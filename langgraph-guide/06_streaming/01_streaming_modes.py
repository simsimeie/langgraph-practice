"""
LangGraph 스트리밍 모드
=========================

목표: LangGraph의 다양한 스트리밍 모드를 이해하고 사용법을 익힌다.

stream() vs invoke():
- invoke(): 그래프가 완전히 끝날 때까지 기다렸다가 최종 State 반환
- stream(): 각 단계의 결과를 실시간으로 이터레이터로 반환

stream_mode 종류:
1. "values"   : 각 단계 후 전체 State를 반환
2. "updates"  : 각 단계에서 변경된 부분만 반환
3. "messages" : 메시지를 토큰 단위로 스트리밍 (LLM 실시간 출력)
4. "debug"    : 모든 내부 이벤트 (디버깅 전용)

언제 어떤 모드를 쓰나:
- UI에서 타이핑 효과 → "messages"
- 진행 바 표시 → "updates"
- 단계별 전체 상태 필요 → "values"
- 버그 추적 → "debug"
"""

import asyncio
import time
from typing import Annotated, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ============================================================
# 샘플 그래프 구성 (스트리밍 테스트용)
# ============================================================

class StreamState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    step_log: Annotated[List[str], list.__add__]  # 각 단계 로그
    final_answer: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)
# streaming=True: LLM이 토큰을 하나씩 생성해서 반환 (stream_mode="messages"에 필요)


def step1_preprocess(state: StreamState) -> dict:
    """1단계: 입력 전처리"""
    time.sleep(0.1)  # 처리 시간 시뮬레이션
    print("  [step1] 전처리 완료")
    return {
        "step_log": ["step1: 전처리 완료"],
    }


def step2_analyze(state: StreamState) -> dict:
    """2단계: 분석"""
    time.sleep(0.1)
    print("  [step2] 분석 완료")
    return {
        "step_log": ["step2: 분석 완료"],
        "messages": [AIMessage(content="분석을 완료했습니다. LLM 응답을 생성합니다.")]
    }


def step3_generate(state: StreamState) -> dict:
    """3단계: LLM으로 응답 생성 (스트리밍 가능)"""
    print("  [step3] LLM 응답 생성 중...")
    messages = [
        SystemMessage(content="당신은 도움이 되는 AI입니다. 간결하게 답변하세요."),
        *state["messages"]
    ]
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "step_log": ["step3: LLM 응답 생성 완료"],
        "final_answer": response.content
    }


def build_stream_graph():
    """스트리밍 테스트용 3단계 그래프"""
    graph_builder = StateGraph(StreamState)
    graph_builder.add_node("step1_preprocess", step1_preprocess)
    graph_builder.add_node("step2_analyze", step2_analyze)
    graph_builder.add_node("step3_generate", step3_generate)

    graph_builder.add_edge(START, "step1_preprocess")
    graph_builder.add_edge("step1_preprocess", "step2_analyze")
    graph_builder.add_edge("step2_analyze", "step3_generate")
    graph_builder.add_edge("step3_generate", END)

    checkpointer = MemorySaver()
    return graph_builder.compile(checkpointer=checkpointer)


# ============================================================
# stream_mode="values" 예제
# ============================================================
# 각 노드 실행 후 전체 State를 반환합니다.
# 각 단계에서 State가 어떻게 변화하는지 추적할 때 유용합니다.

def demo_stream_values(graph):
    """stream_mode='values' - 각 단계의 전체 State"""
    print("\n=== stream_mode='values' 예제 ===")
    print("각 노드 실행 후 전체 State를 출력합니다\n")

    config = {"configurable": {"thread_id": "stream_values"}}
    input_state = {
        "messages": [HumanMessage(content="LangGraph의 스트리밍 기능을 설명해줘")],
        "step_log": [],
        "final_answer": ""
    }

    step = 0
    for state in graph.stream(input_state, config=config, stream_mode="values"):
        step += 1
        print(f"[State {step}]")
        print(f"  메시지 수: {len(state.get('messages', []))}")
        print(f"  단계 로그: {state.get('step_log', [])}")
        print(f"  최종 답변 길이: {len(state.get('final_answer', ''))}자")
        print()


# ============================================================
# stream_mode="updates" 예제
# ============================================================
# 각 노드가 반환한 변경사항만 반환합니다.
# 어떤 노드가 무엇을 변경했는지 추적할 때 유용합니다.

def demo_stream_updates(graph):
    """stream_mode='updates' - 각 노드의 변경사항만"""
    print("\n=== stream_mode='updates' 예제 ===")
    print("각 노드의 변경사항(업데이트)만 출력합니다\n")

    config = {"configurable": {"thread_id": "stream_updates"}}
    input_state = {
        "messages": [HumanMessage(content="Python의 장점을 3가지 알려줘")],
        "step_log": [],
        "final_answer": ""
    }

    for update in graph.stream(input_state, config=config, stream_mode="updates"):
        # update: {노드이름: 변경된_state_딕셔너리}
        for node_name, changes in update.items():
            print(f"[노드: {node_name}]")
            for key, value in changes.items():
                if key == "messages":
                    print(f"  messages: {len(value)}개 메시지 추가")
                elif key == "step_log":
                    print(f"  step_log: {value}")
                elif key == "final_answer":
                    preview = str(value)[:80]
                    print(f"  final_answer: '{preview}...'")
            print()


# ============================================================
# stream_mode="messages" 예제
# ============================================================
# LLM이 생성하는 메시지를 토큰 단위로 스트리밍합니다.
# 채팅 UI에서 타이핑 효과를 구현할 때 사용합니다.
#
# 반환 형식: (message_chunk, metadata)
# - message_chunk: AIMessageChunk (부분 텍스트)
# - metadata: {"langgraph_node": "노드이름", ...}

def demo_stream_messages(graph):
    """stream_mode='messages' - 토큰별 실시간 스트리밍"""
    print("\n=== stream_mode='messages' 예제 ===")
    print("LLM 응답을 토큰 단위로 실시간 출력합니다\n")

    config = {"configurable": {"thread_id": "stream_messages"}}
    input_state = {
        "messages": [HumanMessage(content="FastAPI를 한 문단으로 설명해줘")],
        "step_log": [],
        "final_answer": ""
    }

    print("AI 응답 (실시간 스트리밍):")
    print("-" * 40)

    current_node = None
    for message_chunk, metadata in graph.stream(
        input_state, config=config, stream_mode="messages"
    ):
        # 노드 변경 시 표시
        node_name = metadata.get("langgraph_node", "")
        if node_name != current_node:
            if current_node is not None:
                print()  # 이전 노드 끝
            current_node = node_name
            if node_name:
                print(f"\n[{node_name}에서 생성 중...]")

        # AIMessageChunk의 content를 즉시 출력
        if hasattr(message_chunk, "content") and message_chunk.content:
            print(message_chunk.content, end="", flush=True)

    print("\n" + "-" * 40)


# ============================================================
# stream_mode="debug" 예제
# ============================================================
# 모든 내부 이벤트를 반환합니다.
# 디버깅할 때 유용하지만, 출력이 매우 많습니다.

def demo_stream_debug(graph):
    """stream_mode='debug' - 내부 이벤트 전체"""
    print("\n=== stream_mode='debug' 예제 ===")
    print("내부 이벤트를 출력합니다 (처음 5개만)\n")

    config = {"configurable": {"thread_id": "stream_debug"}}
    input_state = {
        "messages": [HumanMessage(content="안녕")],
        "step_log": [],
        "final_answer": ""
    }

    for i, event in enumerate(graph.stream(
        input_state, config=config, stream_mode="debug"
    )):
        if i >= 5:
            print("...(이하 생략)")
            break
        print(f"[이벤트 {i+1}]")
        print(f"  타입: {event.get('type', '알 수 없음')}")
        if 'payload' in event:
            payload_keys = list(event['payload'].keys()) if isinstance(event['payload'], dict) else []
            print(f"  페이로드 키: {payload_keys}")
        print()


# ============================================================
# 멀티 모드 동시 스트리밍
# ============================================================
# 여러 stream_mode를 동시에 사용하려면 리스트로 전달합니다.

def demo_multi_mode_streaming(graph):
    """여러 stream_mode 동시 사용"""
    print("\n=== 멀티 모드 스트리밍 예제 ===")
    print("values + updates 동시 스트리밍\n")

    config = {"configurable": {"thread_id": "stream_multi"}}
    input_state = {
        "messages": [HumanMessage(content="1+1은?")],
        "step_log": [],
        "final_answer": ""
    }

    # stream_mode를 리스트로 전달하면 각 이벤트가 (mode, data) 튜플로 반환됩니다
    for mode, data in graph.stream(
        input_state,
        config=config,
        stream_mode=["values", "updates"]
    ):
        print(f"모드: {mode}")
        if mode == "updates":
            for node, changes in data.items():
                print(f"  업데이트된 노드: {node}")
        elif mode == "values":
            print(f"  메시지 수: {len(data.get('messages', []))}")
        print()


# ============================================================
# 비동기 스트리밍 (FastAPI용)
# ============================================================

async def demo_async_streaming():
    """
    astream()을 사용한 비동기 스트리밍.
    FastAPI, Django Channels 등 비동기 환경에서 사용합니다.
    """
    print("\n=== 비동기 스트리밍 예제 ===")

    graph = build_stream_graph()
    config = {"configurable": {"thread_id": "async_stream"}}
    input_state = {
        "messages": [HumanMessage(content="asyncio가 뭔가요?")],
        "step_log": [],
        "final_answer": ""
    }

    print("비동기 스트리밍 중...")
    async for update in graph.astream(
        input_state,
        config=config,
        stream_mode="updates"
    ):
        for node, changes in update.items():
            print(f"  [{node}] 업데이트")

    print("비동기 스트리밍 완료")


# ============================================================
# 스트리밍 이벤트 필터링
# ============================================================

def demo_filtered_streaming(graph):
    """특정 노드의 출력만 필터링해서 표시"""
    print("\n=== 스트리밍 필터링 예제 ===")
    print("step3_generate 노드의 출력만 표시\n")

    config = {"configurable": {"thread_id": "stream_filter"}}
    input_state = {
        "messages": [HumanMessage(content="머신러닝을 한 줄로 설명해줘")],
        "step_log": [],
        "final_answer": ""
    }

    print("최종 LLM 응답: ", end="", flush=True)

    for message_chunk, metadata in graph.stream(
        input_state, config=config, stream_mode="messages"
    ):
        # step3_generate 노드의 출력만 필터링
        if metadata.get("langgraph_node") == "step3_generate":
            if hasattr(message_chunk, "content") and message_chunk.content:
                print(message_chunk.content, end="", flush=True)

    print()


# ============================================================
# 실용적인 스트리밍 래퍼
# ============================================================

def stream_with_callback(graph, input_state, config, on_token=None, on_node_complete=None):
    """
    콜백 함수를 사용한 스트리밍 래퍼.
    실제 애플리케이션에서 사용하기 좋은 패턴입니다.

    Args:
        graph: LangGraph 그래프
        input_state: 초기 State
        config: 설정 (thread_id 포함)
        on_token: 각 토큰 생성 시 호출되는 콜백 함수(token_str)
        on_node_complete: 각 노드 완료 시 호출되는 콜백 함수(node_name, changes)
    """
    # 토큰 스트리밍
    for message_chunk, metadata in graph.stream(
        input_state, config=config, stream_mode="messages"
    ):
        if on_token and hasattr(message_chunk, "content") and message_chunk.content:
            on_token(message_chunk.content)

    # 노드 완료 이벤트 (별도 실행 - 실제로는 통합 필요)
    for update in graph.stream(
        input_state,
        {"configurable": {"thread_id": config["configurable"]["thread_id"] + "_2"}},
        stream_mode="updates"
    ):
        if on_node_complete:
            for node_name, changes in update.items():
                on_node_complete(node_name, changes)


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 스트리밍 모드 예제")
    print("=" * 60)
    print("주의: OPENAI_API_KEY가 필요합니다\n")

    try:
        graph = build_stream_graph()

        demo_stream_values(graph)
        demo_stream_updates(graph)
        demo_stream_messages(graph)
        demo_stream_debug(graph)
        demo_filtered_streaming(graph)

        # 비동기 스트리밍 실행
        asyncio.run(demo_async_streaming())

    except Exception as e:
        print(f"\n⚠️ 오류 (API 키 필요): {e}")

    print("\n" + "=" * 60)
    print("stream_mode 요약:")
    print("  'values'   → 각 단계 후 전체 State")
    print("  'updates'  → 각 단계의 변경사항만")
    print("  'messages' → 토큰별 실시간 출력 (타이핑 효과)")
    print("  'debug'    → 모든 내부 이벤트 (디버깅)")
    print("  [list]     → 여러 모드 동시 사용")
    print("  astream()  → 비동기 환경에서 사용")
    print("=" * 60)
