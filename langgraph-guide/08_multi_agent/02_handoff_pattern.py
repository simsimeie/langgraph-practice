"""
LangGraph Handoff 패턴 (핸드오프 패턴)
========================================

목표: 에이전트가 직접 다른 에이전트에게 작업을 넘기는(handoff) 패턴을 구현한다.

Handoff 패턴 vs Supervisor 패턴:
  Supervisor: 중앙 조율자가 모든 작업을 관리 (Top-down)
  Handoff: 에이전트가 자율적으로 다른 에이전트에게 직접 전달 (Peer-to-peer)

Handoff 패턴의 장점:
  - 에이전트가 더 자율적으로 동작
  - 중앙 병목(Supervisor)이 없음
  - 더 유연한 워크플로우 가능

구현 방법:
  - transfer_to_X 도구를 통해 핸드오프
  - Command(goto="에이전트명")를 반환해서 이동
"""

from typing import Annotated, List, Optional, Literal, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 공유 State 정의
# ============================================================

class HandoffState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # 현재 활성 에이전트
    active_agent: str
    # 작업 완료 여부
    task_complete: bool


# ============================================================
# Handoff 도구 정의
# ============================================================
# 각 에이전트는 다른 에이전트로 작업을 넘길 수 있는 도구를 갖습니다.
# 이 도구들은 실제로 실행되지 않고, LLM이 선택하면 라우팅이 발생합니다.

@tool
def transfer_to_technical_agent(reason: str) -> str:
    """기술적인 질문이나 코드 관련 작업을 기술 에이전트에게 전달합니다.

    Args:
        reason: 전달하는 이유 설명
    """
    return f"기술 에이전트에게 전달 중: {reason}"


@tool
def transfer_to_creative_agent(reason: str) -> str:
    """창의적인 작업(글쓰기, 아이디어 생성 등)을 창작 에이전트에게 전달합니다.

    Args:
        reason: 전달하는 이유 설명
    """
    return f"창작 에이전트에게 전달 중: {reason}"


@tool
def transfer_to_analysis_agent(reason: str) -> str:
    """데이터 분석이나 평가 작업을 분석 에이전트에게 전달합니다.

    Args:
        reason: 전달하는 이유 설명
    """
    return f"분석 에이전트에게 전달 중: {reason}"


@tool
def complete_task(summary: str) -> str:
    """현재 에이전트가 작업을 완료했음을 선언합니다.

    Args:
        summary: 완료된 작업 요약
    """
    return f"작업 완료: {summary}"


# ============================================================
# 에이전트 노드 생성 헬퍼
# ============================================================

def create_agent_node(agent_name: str, system_prompt: str, tools_list: list):
    """
    에이전트 노드를 생성하는 팩토리 함수.

    각 에이전트는:
    1. 자신의 전문 도구로 작업 수행
    2. 필요시 다른 에이전트에게 핸드오프
    3. 작업 완료시 complete_task 호출
    """
    llm_with_tools = llm.bind_tools(tools_list)

    def agent_node(state: HandoffState) -> Command:
        """에이전트 노드 함수"""
        print(f"\n[{agent_name}] 실행 중...")

        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"]
        ]

        response = llm_with_tools.invoke(messages)

        # 도구 호출 분석
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"[{agent_name}] 도구 호출: {tool_name}")
            print(f"[{agent_name}] 이유: {tool_args.get('reason', tool_args.get('summary', ''))}")

            # 핸드오프 처리
            if tool_name == "transfer_to_technical_agent":
                next_node = "technical_agent"
            elif tool_name == "transfer_to_creative_agent":
                next_node = "creative_agent"
            elif tool_name == "transfer_to_analysis_agent":
                next_node = "analysis_agent"
            elif tool_name == "complete_task":
                next_node = END
            else:
                next_node = END

            # 도구 실행 결과 메시지 생성
            tool_result = f"핸드오프: {tool_name}({tool_args})"

            # Command 반환: 다음 노드와 State 업데이트를 동시에
            return Command(
                goto=next_node,
                update={
                    "messages": [
                        response,  # AI의 도구 호출 메시지
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"]
                        )
                    ],
                    "active_agent": next_node if next_node != END else agent_name,
                    "task_complete": next_node == END
                }
            )
        else:
            # 도구 호출 없으면 작업 완료
            print(f"[{agent_name}] 직접 응답 생성")
            return Command(
                goto=END,
                update={
                    "messages": [response],
                    "task_complete": True
                }
            )

    return agent_node


# ============================================================
# 각 에이전트 정의
# ============================================================

# 범용 에이전트: 초기 진입점, 적절한 전문 에이전트에게 핸드오프
GENERAL_AGENT_PROMPT = """당신은 범용 AI 어시스턴트입니다.
사용자의 요청을 분석하고 가장 적합한 전문 에이전트에게 전달합니다.

결정 기준:
- 코드/기술 관련: transfer_to_technical_agent
- 글쓰기/창작 관련: transfer_to_creative_agent
- 분석/평가 관련: transfer_to_analysis_agent
- 직접 해결 가능: complete_task

반드시 도구를 사용하세요."""

general_agent_tools = [
    transfer_to_technical_agent,
    transfer_to_creative_agent,
    transfer_to_analysis_agent,
    complete_task
]

# 기술 에이전트
TECHNICAL_AGENT_PROMPT = """당신은 기술 전문가 AI 에이전트입니다.
코드 작성, 기술 설명, 디버깅을 담당합니다.
작업이 완료되면 complete_task 도구를 사용하세요.
창작이 필요하면 transfer_to_creative_agent를 사용하세요."""

technical_agent_tools = [
    transfer_to_creative_agent,
    transfer_to_analysis_agent,
    complete_task
]

# 창작 에이전트
CREATIVE_AGENT_PROMPT = """당신은 창작 전문가 AI 에이전트입니다.
글쓰기, 아이디어 생성, 스토리텔링을 담당합니다.
작업이 완료되면 complete_task 도구를 사용하세요.
분석이 필요하면 transfer_to_analysis_agent를 사용하세요."""

creative_agent_tools = [
    transfer_to_technical_agent,
    transfer_to_analysis_agent,
    complete_task
]

# 분석 에이전트
ANALYSIS_AGENT_PROMPT = """당신은 분석 전문가 AI 에이전트입니다.
데이터 분석, 평가, 비교를 담당합니다.
작업이 완료되면 complete_task 도구를 사용하세요."""

analysis_agent_tools = [
    transfer_to_technical_agent,
    transfer_to_creative_agent,
    complete_task
]


# ============================================================
# 그래프 구성
# ============================================================

def build_handoff_graph():
    """
    핸드오프 패턴 멀티 에이전트 그래프.

    각 에이전트는 Command를 반환해서 다음 노드를 결정합니다.
    에이전트가 직접 다른 에이전트로 이동하므로 중앙 라우터가 불필요합니다.
    """
    builder = StateGraph(HandoffState)

    # 에이전트 노드 생성 및 추가
    builder.add_node(
        "general_agent",
        create_agent_node("general_agent", GENERAL_AGENT_PROMPT, general_agent_tools)
    )
    builder.add_node(
        "technical_agent",
        create_agent_node("technical_agent", TECHNICAL_AGENT_PROMPT, technical_agent_tools)
    )
    builder.add_node(
        "creative_agent",
        create_agent_node("creative_agent", CREATIVE_AGENT_PROMPT, creative_agent_tools)
    )
    builder.add_node(
        "analysis_agent",
        create_agent_node("analysis_agent", ANALYSIS_AGENT_PROMPT, analysis_agent_tools)
    )

    # 시작점은 범용 에이전트
    builder.add_edge(START, "general_agent")

    # 각 에이전트에서 다른 에이전트로의 엣지
    # (Command goto로 동적으로 결정되므로 명시적 엣지 불필요)
    # 하지만 그래프 유효성 검사를 위해 가능한 경로를 선언할 수 있습니다

    return builder.compile(checkpointer=MemorySaver())


# ============================================================
# 실행
# ============================================================

def run_handoff_demo():
    """핸드오프 패턴 실행 예제"""
    print("=== 핸드오프 패턴 멀티 에이전트 ===\n")
    graph = build_handoff_graph()

    test_requests = [
        "Python으로 피보나치 수열을 생성하는 코드를 작성해줘",
        "AI의 미래에 대한 짧은 에세이를 써줘",
        "GPT-4와 Claude의 장단점을 비교 분석해줘",
    ]

    for i, request in enumerate(test_requests[:1], 1):  # API 비용 절약: 첫 번째만
        print(f"[요청 {i}]: {request}")
        config = {"configurable": {"thread_id": f"handoff_demo_{i}"}}

        try:
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=request)],
                    "active_agent": "general_agent",
                    "task_complete": False
                },
                config=config
            )

            print(f"\n활성 에이전트: {result.get('active_agent', '알 수 없음')}")
            print(f"작업 완료: {result.get('task_complete', False)}")

            final_msg = result["messages"][-1]
            if hasattr(final_msg, 'content') and final_msg.content:
                print(f"최종 응답: {final_msg.content[:200]}")

        except Exception as e:
            print(f"⚠️ 오류 (API 키 필요): {e}")

        print()


# ============================================================
# Command 패턴 직접 구현 예제
# ============================================================
# Command 없이 조건부 엣지로도 핸드오프를 구현할 수 있습니다.

class SimpleHandoffState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_agent: str


def triage_node(state: SimpleHandoffState) -> dict:
    """요청을 분류해서 next_agent를 설정"""
    last_msg = state["messages"][-1].content.lower()

    if any(kw in last_msg for kw in ["코드", "python", "함수"]):
        next_agent = "dev_agent"
    elif any(kw in last_msg for kw in ["글", "에세이", "설명"]):
        next_agent = "writer_agent"
    else:
        next_agent = "general_agent_simple"

    print(f"[triage] → {next_agent}")
    return {"next_agent": next_agent}


def dev_agent_simple(state: SimpleHandoffState) -> dict:
    print("[dev_agent] 개발 작업 처리")
    return {"messages": [AIMessage(content="개발 작업 완료!")]}


def writer_agent_simple(state: SimpleHandoffState) -> dict:
    print("[writer_agent] 글쓰기 작업 처리")
    return {"messages": [AIMessage(content="글쓰기 작업 완료!")]}


def general_agent_simple(state: SimpleHandoffState) -> dict:
    print("[general_agent_simple] 일반 작업 처리")
    return {"messages": [AIMessage(content="일반 작업 완료!")]}


def route_to_agent(state: SimpleHandoffState) -> Literal["dev_agent", "writer_agent", "general_agent_simple"]:
    return state["next_agent"]  # type: ignore


def build_simple_handoff_graph():
    """조건부 엣지를 사용한 간단한 핸드오프 예제"""
    builder = StateGraph(SimpleHandoffState)

    builder.add_node("triage", triage_node)
    builder.add_node("dev_agent", dev_agent_simple)
    builder.add_node("writer_agent", writer_agent_simple)
    builder.add_node("general_agent_simple", general_agent_simple)

    builder.add_edge(START, "triage")
    builder.add_conditional_edges("triage", route_to_agent)

    for node in ["dev_agent", "writer_agent", "general_agent_simple"]:
        builder.add_edge(node, END)

    return builder.compile()


if __name__ == "__main__":
    print("=" * 60)
    print("핸드오프 패턴 예제")
    print("=" * 60)

    # 간단한 버전 (API 키 불필요)
    print("\n[간단한 핸드오프 - 조건부 엣지 방식]")
    simple_graph = build_simple_handoff_graph()

    for request in ["Python 코드 작성해줘", "에세이 써줘", "안녕하세요"]:
        print(f"\n요청: '{request}'")
        result = simple_graph.invoke({
            "messages": [HumanMessage(content=request)],
            "next_agent": ""
        })
        print(f"응답: {result['messages'][-1].content}")

    # Command 기반 버전 (API 키 필요)
    print("\n" + "-" * 60)
    print("[Command 기반 핸드오프 - API 키 필요]")
    run_handoff_demo()

    print("\n" + "=" * 60)
    print("핸드오프 패턴 핵심:")
    print("1. 각 에이전트가 다음 에이전트를 직접 결정")
    print("2. Command(goto=에이전트명)으로 이동")
    print("3. transfer_to_X 도구로 LLM이 핸드오프 결정")
    print("4. 중앙 수퍼바이저 없이 peer-to-peer 협력")
    print("=" * 60)
