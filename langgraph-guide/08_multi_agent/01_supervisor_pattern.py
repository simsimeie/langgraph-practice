"""
LangGraph Supervisor 패턴 (수퍼바이저 패턴)
============================================

목표: 중앙 수퍼바이저가 여러 전문 에이전트를 조율하는 멀티 에이전트 시스템을 구현한다.

수퍼바이저 패턴:
  1. 사용자 요청이 수퍼바이저에게 도달
  2. 수퍼바이저가 작업을 분석하고 적절한 에이전트에게 위임
  3. 해당 에이전트가 작업 수행 후 수퍼바이저에게 결과 보고
  4. 수퍼바이저가 충분한지 판단하고, 필요시 다른 에이전트에게 추가 작업 요청
  5. 모든 작업 완료 시 최종 응답 생성

에이전트 구성:
  - 수퍼바이저 (Supervisor): 작업 조율
  - 연구원 에이전트 (Researcher): 정보 수집/분석
  - 코더 에이전트 (Coder): 코드 작성/실행
  - 검토자 에이전트 (Reviewer): 결과 검증
"""

from typing import Annotated, List, Literal, TypedDict
import operator
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 공유 State 정의
# ============================================================

class MultiAgentState(TypedDict):
    """모든 에이전트가 공유하는 State"""
    messages: Annotated[List[BaseMessage], add_messages]
    # 현재 작업 내용
    current_task: str
    # 수퍼바이저가 결정한 다음 에이전트
    next_agent: str
    # 각 에이전트의 작업 결과
    agent_outputs: Annotated[List[str], operator.add]
    # 작업 완료 여부
    is_complete: bool
    # 최종 응답
    final_response: str


# ============================================================
# 도구 정의 (각 에이전트가 사용할 도구들)
# ============================================================

@tool
def research_topic(topic: str) -> str:
    """특정 주제를 연구하고 정보를 수집합니다.

    Args:
        topic: 연구할 주제
    """
    print(f"    [research_tool] '{topic}' 연구 중...")
    # 시뮬레이션
    return f"'{topic}'에 대한 연구 결과: 핵심 개념, 최신 트렌드, 주요 사례를 수집했습니다."


@tool
def write_code(language: str, requirements: str) -> str:
    """요구사항에 맞는 코드를 작성합니다.

    Args:
        language: 프로그래밍 언어
        requirements: 코드 요구사항
    """
    print(f"    [code_tool] {language} 코드 작성 중...")
    code = f"""# {requirements}
def solution():
    # 자동 생성된 {language} 코드
    print("요구사항 구현 완료")
    return True
"""
    return f"코드 작성 완료:\n```{language}\n{code}\n```"


@tool
def review_content(content: str, criteria: str) -> str:
    """콘텐츠를 지정된 기준으로 검토합니다.

    Args:
        content: 검토할 콘텐츠
        criteria: 검토 기준
    """
    print(f"    [review_tool] 기준 '{criteria}'으로 검토 중...")
    # 시뮬레이션: 무작위로 통과/보완 결정
    import random
    if random.random() > 0.3:
        return f"검토 완료: '{criteria}' 기준 충족. 품질 양호합니다."
    else:
        return f"검토 결과: '{criteria}' 일부 보완 필요. 구체성 향상 권장."


# ============================================================
# 각 에이전트 노드 정의
# ============================================================

SUPERVISOR_PROMPT = """당신은 멀티 에이전트 팀의 수퍼바이저입니다.
팀 구성:
- researcher: 정보 수집과 분석 담당
- coder: 코드 작성과 기술 구현 담당
- reviewer: 결과물 검토와 품질 보증 담당
- FINISH: 모든 작업이 완료되어 최종 응답을 생성할 준비가 된 경우

현재까지의 진행 상황을 보고 다음에 어떤 에이전트를 실행할지 결정하세요.
반드시 다음 중 하나만 답하세요: researcher, coder, reviewer, FINISH"""


def supervisor_node(state: MultiAgentState) -> dict:
    """
    수퍼바이저 노드: 다음 실행할 에이전트를 결정합니다.

    수퍼바이저는 현재 상황을 분석하고:
    - 아직 처리되지 않은 작업이 있으면 해당 에이전트 선택
    - 충분한 결과가 모이면 FINISH 결정
    """
    print("\n[supervisor] 다음 에이전트 결정 중...")

    messages_for_supervisor = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        *state["messages"]
    ]

    # LLM으로 다음 에이전트 결정
    # 실제로는 Structured Output으로 더 안정적으로 구현
    response = llm.invoke(messages_for_supervisor)
    decision = response.content.strip().lower()

    # 유효한 선택지로 정규화
    valid_choices = ["researcher", "coder", "reviewer", "finish"]
    next_agent = "finish"  # 기본값

    for choice in valid_choices:
        if choice in decision:
            next_agent = choice
            break

    print(f"[supervisor] 결정: {next_agent}")

    return {
        "next_agent": next_agent,
        "messages": [AIMessage(content=f"수퍼바이저: 다음 작업은 '{next_agent}'에게 위임합니다.")]
    }


RESEARCHER_PROMPT = """당신은 전문 연구원 에이전트입니다.
주어진 주제에 대해 깊이 있는 정보를 수집하고 분석합니다.
research_topic 도구를 사용하세요."""

llm_with_research_tool = llm.bind_tools([research_topic])

def researcher_node(state: MultiAgentState) -> dict:
    """연구원 에이전트: 정보를 수집하고 분석합니다."""
    print("[researcher] 연구 시작...")

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        *state["messages"]
    ]

    response = llm_with_research_tool.invoke(messages)

    # 도구 호출 시뮬레이션
    if response.tool_calls:
        tool_name = response.tool_calls[0]["name"]
        tool_args = response.tool_calls[0]["args"]
        result = research_topic.invoke(tool_args)
        output = f"[연구원 완료] {result}"
    else:
        output = f"[연구원 완료] {response.content}"

    print(f"[researcher] {output[:80]}...")
    return {
        "agent_outputs": [output],
        "messages": [AIMessage(content=output)]
    }


CODER_PROMPT = """당신은 전문 소프트웨어 엔지니어 에이전트입니다.
고품질의 코드를 작성하고 기술적 문제를 해결합니다.
write_code 도구를 사용하세요."""

llm_with_code_tool = llm.bind_tools([write_code])

def coder_node(state: MultiAgentState) -> dict:
    """코더 에이전트: 코드를 작성합니다."""
    print("[coder] 코드 작성 시작...")

    messages = [
        SystemMessage(content=CODER_PROMPT),
        *state["messages"]
    ]

    response = llm_with_code_tool.invoke(messages)

    if response.tool_calls:
        tool_args = response.tool_calls[0]["args"]
        result = write_code.invoke(tool_args)
        output = f"[코더 완료] {result}"
    else:
        output = f"[코더 완료] {response.content}"

    print(f"[coder] 코드 작성 완료")
    return {
        "agent_outputs": [output],
        "messages": [AIMessage(content=output)]
    }


REVIEWER_PROMPT = """당신은 전문 검토자 에이전트입니다.
생성된 내용의 품질을 검토하고 개선점을 제안합니다.
review_content 도구를 사용하세요."""

llm_with_review_tool = llm.bind_tools([review_content])

def reviewer_node(state: MultiAgentState) -> dict:
    """검토자 에이전트: 결과물을 검토합니다."""
    print("[reviewer] 검토 시작...")

    messages = [
        SystemMessage(content=REVIEWER_PROMPT),
        *state["messages"]
    ]

    response = llm_with_review_tool.invoke(messages)

    if response.tool_calls:
        tool_args = response.tool_calls[0]["args"]
        result = review_content.invoke(tool_args)
        output = f"[검토자 완료] {result}"
    else:
        output = f"[검토자 완료] {response.content}"

    print(f"[reviewer] 검토 완료")
    return {
        "agent_outputs": [output],
        "messages": [AIMessage(content=output)]
    }


def finalize_node(state: MultiAgentState) -> dict:
    """최종 응답을 생성하는 노드"""
    print("[finalize] 최종 응답 생성...")

    outputs = state["agent_outputs"]
    final_response = f"작업 완료 요약:\n" + "\n".join(f"- {o[:100]}" for o in outputs)

    return {
        "final_response": final_response,
        "is_complete": True,
        "messages": [AIMessage(content=final_response)]
    }


# ============================================================
# 라우팅 함수
# ============================================================

def route_supervisor(state: MultiAgentState) -> Literal["researcher", "coder", "reviewer", "finalize"]:
    """수퍼바이저의 결정에 따라 다음 노드 선택"""
    next_agent = state.get("next_agent", "finish")

    # 최대 반복 방지 (실제로는 iteration_count 같은 필드 사용)
    agent_output_count = len(state.get("agent_outputs", []))
    if agent_output_count >= 3 or next_agent == "finish":
        return "finalize"

    routes = {
        "researcher": "researcher",
        "coder": "coder",
        "reviewer": "reviewer",
    }
    return routes.get(next_agent, "finalize")


# ============================================================
# 그래프 구성
# ============================================================

def build_supervisor_graph():
    """
    수퍼바이저 패턴 멀티 에이전트 그래프.

    구조:
    START
      │
      ▼
    [supervisor] ──▶ researcher ──┐
         │                         │
         ├──▶ coder ────────────────┤ 결과를 supervisor에게 보고
         │                         │
         ├──▶ reviewer ─────────────┘
         │
         └──▶ finalize → END
    """
    builder = StateGraph(MultiAgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "supervisor")

    # 수퍼바이저에서 각 에이전트로 조건부 라우팅
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "coder": "coder",
            "reviewer": "reviewer",
            "finalize": "finalize"
        }
    )

    # 각 에이전트 완료 후 수퍼바이저로 돌아옴
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("coder", "supervisor")
    builder.add_edge("reviewer", "supervisor")

    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=MemorySaver())


# ============================================================
# 실행
# ============================================================

def run_supervisor_demo():
    """수퍼바이저 패턴 실행 예제"""
    print("=== 수퍼바이저 멀티 에이전트 시스템 ===\n")
    graph = build_supervisor_graph()

    config = {"configurable": {"thread_id": "supervisor_demo"}}

    request = """LangGraph를 사용한 챗봇 개발 프로젝트가 필요합니다.
1. LangGraph에 대해 연구해주세요
2. 기본 챗봇 코드를 작성해주세요
3. 작성된 코드를 검토해주세요"""

    print(f"요청: {request}\n")
    print("=" * 50)

    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=request)],
                "current_task": request,
                "next_agent": "",
                "agent_outputs": [],
                "is_complete": False,
                "final_response": ""
            },
            config=config
        )

        print("\n" + "=" * 50)
        print("최종 결과:")
        print(result["final_response"])
        print(f"\n총 에이전트 출력: {len(result['agent_outputs'])}개")

    except Exception as e:
        print(f"⚠️ 오류 (API 키 필요): {e}")


if __name__ == "__main__":
    run_supervisor_demo()

    print("\n" + "=" * 60)
    print("수퍼바이저 패턴 핵심 요소:")
    print("1. 수퍼바이저가 작업을 분석하고 에이전트에게 위임")
    print("2. 각 에이전트는 완료 후 수퍼바이저에게 보고")
    print("3. 수퍼바이저가 FINISH를 결정하면 최종 응답 생성")
    print("4. 무한 루프 방지: iteration_count 같은 제한 필요")
    print("=" * 60)
