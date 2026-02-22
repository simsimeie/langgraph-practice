"""
LangGraph ReAct 에이전트 구현
================================

목표: ReAct(Reasoning + Acting) 패턴의 에이전트를 직접 구현한다.

ReAct 패턴이란?
  LLM이 문제를 해결할 때 다음 사이클을 반복합니다:
  1. Thought (추론): 현재 상황을 분석하고 무엇을 해야 할지 생각
  2. Action (행동): 도구를 호출하거나 최종 답변 결정
  3. Observation (관찰): 도구 실행 결과 확인
  → 다시 1로 돌아가거나 최종 답변 출력

내장 prebuilt 에이전트 vs 직접 구현:
- create_react_agent(): LangGraph가 제공하는 완성된 ReAct 에이전트
- 직접 구현: 커스터마이징이 필요할 때 사용
"""

from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent

load_dotenv()


# ============================================================
# 도구 정의
# ============================================================

@tool
def search_web(query: str) -> str:
    """웹에서 최신 정보를 검색합니다.

    Args:
        query: 검색 쿼리

    Returns:
        검색 결과 요약
    """
    # 실제로는 Tavily, SerpAPI 등의 검색 API 사용
    mock_results = {
        "LangGraph": "LangGraph는 LLM 기반 에이전트 워크플로우를 위한 프레임워크입니다.",
        "파이썬": "Python은 간결한 문법을 가진 범용 프로그래밍 언어입니다.",
        "인공지능": "AI는 기계가 인간처럼 학습하고 추론하는 기술입니다.",
    }

    for key, value in mock_results.items():
        if key.lower() in query.lower():
            result = value
            break
    else:
        result = f"'{query}'에 대한 검색 결과: 관련 정보를 찾았습니다."

    print(f"[search_web] 쿼리: '{query}' → 결과 반환")
    return result


@tool
def get_stock_price(symbol: str) -> str:
    """주식 종목의 현재 가격을 조회합니다.

    Args:
        symbol: 주식 심볼 (예: AAPL, GOOGL, TSLA)

    Returns:
        현재 주가 정보
    """
    # 시뮬레이션 데이터
    mock_prices = {
        "AAPL": {"price": 185.50, "change": "+1.2%"},
        "GOOGL": {"price": 175.30, "change": "-0.5%"},
        "TSLA": {"price": 220.15, "change": "+3.1%"},
        "NVDA": {"price": 875.00, "change": "+5.2%"},
    }
    symbol_upper = symbol.upper()
    if symbol_upper in mock_prices:
        data = mock_prices[symbol_upper]
        result = f"{symbol_upper}: ${data['price']} ({data['change']})"
    else:
        result = f"{symbol}: 데이터를 찾을 수 없습니다."

    print(f"[get_stock_price] {symbol}: {result}")
    return result


@tool
def run_python_code(code: str) -> str:
    """간단한 Python 코드를 실행하고 결과를 반환합니다.

    Args:
        code: 실행할 Python 코드 (단일 표현식 또는 간단한 계산)

    Returns:
        코드 실행 결과
    """
    # 보안상 eval은 실제로는 위험합니다.
    # 실제 환경에서는 샌드박스를 사용하세요.
    try:
        # 안전한 계산만 허용 (숫자, 기본 수학 연산)
        allowed_names = {
            "__builtins__": {},
            "abs": abs, "round": round, "max": max, "min": min,
            "sum": sum, "len": len, "int": int, "float": float,
            "pow": pow,
        }
        import math
        allowed_names.update({k: getattr(math, k) for k in dir(math) if not k.startswith('_')})

        result = eval(code, allowed_names)  # noqa: S307 (교육용 예시)
        print(f"[run_python_code] '{code}' → {result}")
        return str(result)
    except Exception as e:
        return f"실행 오류: {str(e)}"


tools = [search_web, get_stock_price, run_python_code]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 방법 1: create_react_agent() 사용 (권장)
# ============================================================
# LangGraph가 제공하는 완성된 ReAct 에이전트입니다.
# 대부분의 경우 이것으로 충분합니다.

def build_prebuilt_agent(system_prompt: str = None):
    """
    내장 create_react_agent()로 에이전트 생성.

    이 함수는 내부적으로:
    1. StateGraph 생성
    2. agent 노드 + tools 노드 추가
    3. tools_condition으로 조건부 엣지 추가
    4. 컴파일
    를 자동으로 처리합니다.
    """
    if system_prompt:
        # 시스템 프롬프트를 포함한 에이전트
        return create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt  # LangGraph 1.x에서는 prompt 파라미터 사용
        )
    else:
        return create_react_agent(model=llm, tools=tools)


# ============================================================
# 방법 2: 직접 구현
# ============================================================
# 커스터마이징이 필요한 경우 직접 구현합니다.
# 예: 중간 단계 로깅, 특수 라우팅 로직, 최대 반복 횟수 제한

class ReActState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # 현재 반복 횟수 추적 (무한루프 방지)
    iteration_count: int
    # 최대 반복 허용 횟수
    max_iterations: int


llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """당신은 도움이 되는 AI 어시스턴트입니다.
주어진 도구들을 활용해서 사용자의 질문에 답하세요.
복잡한 질문은 단계별로 도구를 사용해서 해결하세요."""


def agent_node(state: ReActState) -> dict:
    """에이전트 노드 - LLM이 다음 행동 결정"""
    iteration = state["iteration_count"]
    print(f"\n[agent] 반복 #{iteration + 1}")

    # 시스템 메시지 + 현재 대화 기록
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        print(f"[agent] 도구 호출 결정: {tool_names}")
    else:
        print(f"[agent] 최종 답변 생성")

    return {
        "messages": [response],
        "iteration_count": iteration + 1
    }


def route_agent(state: ReActState) -> Literal["tools", "__end__"]:
    """
    도구 호출 여부와 최대 반복 횟수를 체크하는 라우터.

    tools_condition과 유사하지만 최대 반복 횟수 제한이 추가됩니다.
    """
    last_message = state["messages"][-1]

    # 최대 반복 횟수 초과 시 강제 종료
    if state["iteration_count"] >= state["max_iterations"]:
        print(f"[라우터] 최대 반복 횟수 초과 → 강제 종료")
        return "__end__"

    # 도구 호출이 있으면 tools 노드로
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 도구 호출 없으면 종료
    return "__end__"


def build_custom_react_agent(max_iterations: int = 10):
    """커스텀 ReAct 에이전트 그래프"""
    graph_builder = StateGraph(ReActState)

    graph_builder.add_node("agent", agent_node)

    # ToolNode: 도구 실행 담당
    # handle_tool_errors=True: 도구 실행 중 오류가 발생해도 계속 진행
    tool_node = ToolNode(tools, handle_tool_errors=True)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "agent")

    # 커스텀 라우터 사용
    graph_builder.add_conditional_edges("agent", route_agent)
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile()


# ============================================================
# 멀티스텝 작업 예제
# ============================================================
# ReAct 에이전트가 진가를 발휘하는 상황: 여러 단계가 필요한 질문

MULTI_STEP_EXAMPLES = [
    # 단일 도구 사용
    "AAPL 주가가 얼마예요?",
    # 계산이 필요한 경우
    "2의 10제곱은 얼마예요?",
    # 정보 검색
    "LangGraph가 무엇인지 검색해줘",
    # 복합적인 질문 (여러 도구 사용 가능)
    "TSLA와 NVDA 주가를 각각 알려주고 차이도 계산해줘",
]


def run_multi_step_demo():
    """멀티스텝 에이전트 실행 예제"""
    print("\n=== 멀티스텝 ReAct 에이전트 예제 ===")

    try:
        agent = build_custom_react_agent(max_iterations=5)

        for question in MULTI_STEP_EXAMPLES[:2]:  # API 비용 절약을 위해 처음 2개만
            print(f"\n{'='*50}")
            print(f"질문: {question}")
            print("-" * 50)

            result = agent.invoke({
                "messages": [HumanMessage(content=question)],
                "iteration_count": 0,
                "max_iterations": 5
            })

            final_answer = result["messages"][-1].content
            print(f"\n최종 답변: {final_answer}")
            print(f"총 반복 횟수: {result['iteration_count']}")

    except Exception as e:
        print(f"⚠️ 실행 오류 (API 키 필요): {e}")


# ============================================================
# 에이전트 스트리밍 실행
# ============================================================

def stream_agent_steps():
    """에이전트의 각 단계를 스트리밍으로 출력"""
    print("\n=== 에이전트 단계별 스트리밍 ===")

    try:
        agent = build_custom_react_agent()

        print("질문: AAPL 주가를 알려주고, 그 가격의 제곱근을 계산해줘")
        print("-" * 50)

        # stream()으로 각 단계의 state 변화를 실시간으로 확인
        for step, event in enumerate(agent.stream(
            {
                "messages": [HumanMessage(content="AAPL 주가를 알려주고, 그 가격의 제곱근을 계산해줘")],
                "iteration_count": 0,
                "max_iterations": 5
            },
            stream_mode="updates"  # 각 노드의 업데이트만 반환
        )):
            print(f"\n[단계 {step + 1}]")
            for node_name, node_output in event.items():
                print(f"  노드: {node_name}")
                if "messages" in node_output:
                    last_msg = node_output["messages"][-1]
                    msg_type = type(last_msg).__name__
                    content_preview = str(last_msg.content)[:100]
                    print(f"  메시지 타입: {msg_type}")
                    print(f"  내용: {content_preview}...")

    except Exception as e:
        print(f"⚠️ 실행 오류 (API 키 필요): {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph ReAct 에이전트 예제")
    print("=" * 60)

    # 도구 확인
    print("\n사용 가능한 도구:")
    for t in tools:
        print(f"  - {t.name}: {t.description[:60]}...")

    # 도구 직접 테스트
    print("\n=== 도구 직접 테스트 ===")
    print(search_web.invoke({"query": "LangGraph"}))
    print(get_stock_price.invoke({"symbol": "AAPL"}))
    print(run_python_code.invoke({"code": "sum(range(1, 101))"}))

    # 에이전트 실행 (API 키 필요)
    run_multi_step_demo()
    stream_agent_steps()
