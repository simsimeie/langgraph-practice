"""
LangGraph 병렬 노드 실행 (Send API)
=====================================

목표: Send API를 사용해서 여러 노드를 동시에 병렬로 실행하는 방법을 이해한다.

병렬 실행이 유용한 경우:
- 여러 독립적인 API를 동시에 호출할 때
- 여러 문서를 동시에 처리할 때
- 팬아웃(Fan-out): 하나의 작업을 여러 병렬 작업으로 분배

핵심 API: Send
- from langgraph.types import Send
- Send(node_name, state): 특정 노드에 특정 state를 보내는 이벤트
- 여러 Send를 반환하면 동시에 병렬로 실행됩니다

기본 병렬 실행 vs Send API:
- 기본: 여러 노드에 같은 엣지를 추가하면 순차 실행
- Send API: 다른 state를 각 노드에 보내서 진정한 병렬 실행
"""

import asyncio
import time
from typing import Annotated, List, TypedDict
import operator
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 패턴 1: 기본 병렬 실행 (Fan-Out)
# ============================================================
# 하나의 노드에서 여러 노드로 동시에 분기합니다.
# 모든 병렬 노드가 완료되면 다음 노드로 합류(Fan-In)합니다.

class FanOutState(TypedDict):
    query: str
    # operator.add: 각 병렬 노드의 결과가 리스트에 추가됩니다
    search_results: Annotated[List[str], operator.add]
    final_summary: str


def search_database_node(state: FanOutState) -> dict:
    """데이터베이스 검색 (병렬 실행)"""
    time.sleep(0.1)  # 비동기 작업 시뮬레이션
    result = f"DB 검색 결과: '{state['query']}'와 관련된 항목 3개 발견"
    print(f"  [search_database] 완료: {result[:50]}")
    return {"search_results": [result]}


def search_web_node(state: FanOutState) -> dict:
    """웹 검색 (병렬 실행)"""
    time.sleep(0.15)
    result = f"웹 검색 결과: '{state['query']}'에 대한 웹페이지 5개 발견"
    print(f"  [search_web] 완료: {result[:50]}")
    return {"search_results": [result]}


def search_docs_node(state: FanOutState) -> dict:
    """문서 검색 (병렬 실행)"""
    time.sleep(0.08)
    result = f"문서 검색 결과: '{state['query']}'와 관련된 문서 2개 발견"
    print(f"  [search_docs] 완료: {result[:50]}")
    return {"search_results": [result]}


def summarize_results_node(state: FanOutState) -> dict:
    """모든 검색 결과를 합쳐서 요약 (Fan-In)"""
    print(f"[summarize] {len(state['search_results'])}개 결과 수집 완료")
    summary = "통합 검색 결과:\n" + "\n".join(
        f"  {i+1}. {r}" for i, r in enumerate(state["search_results"])
    )
    return {"final_summary": summary}


def build_fanout_graph():
    """
    팬아웃-팬인 그래프.

    START → [search_database, search_web, search_docs] 동시 실행
         → summarize (모든 검색 완료 후)

    주의: LangGraph에서 병렬 실행은 여러 엣지를 통해 자동 처리됩니다.
    여러 노드가 같은 키에 값을 쓸 때 Reducer가 합쳐줍니다.
    """
    builder = StateGraph(FanOutState)

    builder.add_node("search_database", search_database_node)
    builder.add_node("search_web", search_web_node)
    builder.add_node("search_docs", search_docs_node)
    builder.add_node("summarize", summarize_results_node)

    # 모든 검색 노드를 START에서 동시에 시작
    builder.add_edge(START, "search_database")
    builder.add_edge(START, "search_web")
    builder.add_edge(START, "search_docs")

    # 모든 검색 노드에서 summarize로 (모두 완료 후 실행됨)
    builder.add_edge("search_database", "summarize")
    builder.add_edge("search_web", "summarize")
    builder.add_edge("search_docs", "summarize")

    builder.add_edge("summarize", END)

    return builder.compile()


# ============================================================
# 패턴 2: Send API를 사용한 동적 병렬 처리
# ============================================================
# Send는 런타임에 동적으로 병렬 작업을 생성합니다.
# 처리할 항목 수가 실행 전에 알 수 없을 때 사용합니다.
#
# 예: 문서 목록이 동적으로 결정될 때, 각 문서를 병렬로 처리

class DocumentState(TypedDict):
    """개별 문서 처리를 위한 State"""
    document: str    # 처리할 문서 내용
    doc_id: int      # 문서 ID


class DocumentProcessingState(TypedDict):
    """문서 처리 메인 State"""
    documents: List[str]         # 처리할 문서 목록
    # 각 문서 처리 결과가 여기에 모임
    summaries: Annotated[List[str], operator.add]
    final_report: str


def distribute_documents_node(state: DocumentProcessingState) -> List[Send]:
    """
    문서 목록을 받아서 각 문서를 병렬로 처리하는 Send 이벤트 생성.

    이 함수는 일반 딕셔너리 대신 List[Send]를 반환합니다.
    LangGraph는 Send 목록을 보고 동시에 여러 노드를 실행합니다.
    """
    documents = state["documents"]
    print(f"[distribute] {len(documents)}개 문서를 병렬 처리 시작")

    # 각 문서에 대해 Send 이벤트 생성
    # Send(노드이름, 해당_노드에_전달할_state)
    sends = [
        Send(
            "process_document",  # 실행할 노드 이름
            {                    # 해당 노드에 전달할 State
                "document": doc,
                "doc_id": i
            }
        )
        for i, doc in enumerate(documents)
    ]

    return sends  # Send 목록 반환 → 병렬 실행


def process_document_node(state: DocumentState) -> dict:
    """
    개별 문서를 처리하는 노드 (병렬로 여러 번 실행됨).

    각 Send에 의해 독립적으로 실행됩니다.
    """
    doc_id = state["doc_id"]
    doc = state["document"]
    time.sleep(0.1 * (doc_id + 1))  # 각 문서마다 다른 처리 시간

    # 실제로는 LLM으로 요약
    summary = f"[문서 {doc_id}] 요약: '{doc[:30]}...' (키워드: LangGraph, AI)"
    print(f"  [process_document] 문서 {doc_id} 처리 완료")

    # process_document의 결과는 DocumentProcessingState의 summaries에 추가됨
    # (Reducer가 처리)
    return {"summaries": [summary]}


def generate_report_node(state: DocumentProcessingState) -> dict:
    """모든 문서 요약을 합쳐서 최종 보고서 생성"""
    print(f"[generate_report] {len(state['summaries'])}개 요약 통합")
    report = "=== 최종 보고서 ===\n"
    for i, summary in enumerate(state["summaries"], 1):
        report += f"\n{i}. {summary}"
    return {"final_report": report}


def build_send_parallel_graph():
    """
    Send API를 사용한 동적 병렬 처리 그래프.

    START → distribute → (동적으로 생성된 process_document × N) → generate_report → END
    """
    builder = StateGraph(DocumentProcessingState)

    builder.add_node("distribute", distribute_documents_node)
    builder.add_node("process_document", process_document_node)
    builder.add_node("generate_report", generate_report_node)

    builder.add_edge(START, "distribute")

    # distribute 노드에서 반환된 Send들이 process_document를 병렬 실행
    # process_document가 모두 완료되면 generate_report 실행
    builder.add_edge("process_document", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()


# ============================================================
# 패턴 3: 조건부 병렬 처리
# ============================================================
# 조건에 따라 다른 수의 병렬 작업을 생성합니다.

class ConditionalParallelState(TypedDict):
    input_data: str
    priority: str   # "high" | "low"
    results: Annotated[List[str], operator.add]
    output: str


def conditional_distribute_node(state: ConditionalParallelState) -> List[Send]:
    """우선순위에 따라 처리 노드 수를 동적으로 결정"""
    if state["priority"] == "high":
        # 고우선순위: 3개 노드에서 동시 처리
        print("[distribute] 고우선순위: 3개 노드 병렬 실행")
        return [
            Send("fast_process", {"data": state["input_data"], "worker_id": i})
            for i in range(3)
        ]
    else:
        # 저우선순위: 1개 노드만
        print("[distribute] 저우선순위: 1개 노드 순차 실행")
        return [Send("fast_process", {"data": state["input_data"], "worker_id": 0})]


class WorkerState(TypedDict):
    data: str
    worker_id: int


def fast_process_node(state: WorkerState) -> dict:
    """작업 처리 노드"""
    result = f"Worker {state['worker_id']} 처리 결과: {state['data'][:20]}"
    print(f"  [fast_process] {result[:50]}")
    return {"results": [result]}


def merge_results_node(state: ConditionalParallelState) -> dict:
    output = " | ".join(state["results"])
    return {"output": output}


def build_conditional_parallel_graph():
    builder = StateGraph(ConditionalParallelState)
    builder.add_node("conditional_distribute", conditional_distribute_node)
    builder.add_node("fast_process", fast_process_node)
    builder.add_node("merge_results", merge_results_node)

    builder.add_edge(START, "conditional_distribute")
    builder.add_edge("fast_process", "merge_results")
    builder.add_edge("merge_results", END)

    return builder.compile()


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("병렬 노드 실행 예제")
    print("=" * 60)

    # 패턴 1: 팬아웃-팬인
    print("\n[패턴 1: 팬아웃-팬인 (Fan-Out/Fan-In)]")
    fanout_graph = build_fanout_graph()
    result = fanout_graph.invoke({
        "query": "LangGraph 사용법",
        "search_results": [],
        "final_summary": ""
    })
    print(f"\n최종 요약:\n{result['final_summary']}")

    # 패턴 2: Send API
    print("\n" + "-" * 60)
    print("[패턴 2: Send API 동적 병렬 처리]")
    docs = [
        "LangGraph는 상태 기반 AI 에이전트를 구현하는 프레임워크입니다.",
        "LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.",
        "Python은 AI 개발에서 가장 인기 있는 프로그래밍 언어입니다.",
    ]

    send_graph = build_send_parallel_graph()
    result = send_graph.invoke({
        "documents": docs,
        "summaries": [],
        "final_report": ""
    })
    print(f"\n최종 보고서:\n{result['final_report']}")

    # 패턴 3: 조건부 병렬
    print("\n" + "-" * 60)
    print("[패턴 3: 조건부 병렬 처리]")
    cond_graph = build_conditional_parallel_graph()

    for priority in ["high", "low"]:
        print(f"\n우선순위: {priority}")
        result = cond_graph.invoke({
            "input_data": "중요한 데이터 처리",
            "priority": priority,
            "results": [],
            "output": ""
        })
        print(f"결과: {result['output'][:100]}")
