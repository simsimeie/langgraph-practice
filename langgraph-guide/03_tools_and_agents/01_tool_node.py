"""
LangGraph ToolNode 사용법
==========================

목표: @tool 데코레이터로 도구를 만들고 ToolNode로 실행하는 방법을 이해한다.

핵심 개념:
- @tool 데코레이터: Python 함수를 LangChain 도구로 변환
- ToolNode: 도구 실행을 자동화하는 내장 노드
- ToolMessage: 도구 실행 결과를 담는 메시지 타입
- bind_tools(): LLM에 도구 목록을 알려주는 메서드
- tools_condition: 도구 호출 여부를 자동 판단하는 내장 라우터

도구의 작동 방식:
1. LLM이 AIMessage에 tool_calls 필드를 포함해서 응답
2. ToolNode가 tool_calls를 보고 해당 도구를 실행
3. 실행 결과를 ToolMessage로 State에 추가
4. LLM이 ToolMessage를 보고 최종 응답 생성
"""

import math
import json
from typing import Annotated, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ============================================================
# STEP 1: 도구(Tool) 정의
# ============================================================
# @tool 데코레이터를 사용하면 일반 Python 함수를 LLM이 호출할 수 있는
# 도구로 변환합니다.
#
# 중요: docstring이 도구의 설명이 됩니다.
#       LLM은 docstring을 보고 언제 이 도구를 써야 할지 결정합니다.
#       따라서 명확하고 자세한 docstring을 작성해야 합니다.

@tool
def add_numbers(a: float, b: float) -> float:
    """두 숫자를 더합니다. 덧셈 계산이 필요할 때 사용하세요.

    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자

    Returns:
        두 숫자의 합
    """
    result = a + b
    print(f"[add_numbers] {a} + {b} = {result}")
    return result


@tool
def multiply_numbers(a: float, b: float) -> float:
    """두 숫자를 곱합니다. 곱셈 계산이 필요할 때 사용하세요.

    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자

    Returns:
        두 숫자의 곱
    """
    result = a * b
    print(f"[multiply_numbers] {a} × {b} = {result}")
    return result


@tool
def calculate_circle_area(radius: float) -> float:
    """원의 넓이를 계산합니다. 원의 반지름이 주어졌을 때 넓이를 구합니다.

    Args:
        radius: 원의 반지름 (양수)

    Returns:
        원의 넓이 (π × r²)
    """
    if radius <= 0:
        return 0.0
    area = math.pi * radius ** 2
    print(f"[calculate_circle_area] 반지름 {radius} → 넓이 {area:.4f}")
    return round(area, 4)


@tool
def get_weather(city: str) -> str:
    """특정 도시의 날씨 정보를 조회합니다.

    Args:
        city: 날씨를 조회할 도시 이름 (예: 서울, 부산, 뉴욕)

    Returns:
        해당 도시의 현재 날씨 정보
    """
    # 실제로는 날씨 API를 호출합니다.
    # 여기서는 시뮬레이션 데이터를 반환합니다.
    weather_data = {
        "서울": "맑음, 기온 22°C, 습도 60%",
        "부산": "흐림, 기온 25°C, 습도 75%",
        "뉴욕": "비, 기온 15°C, 습도 85%",
        "도쿄": "맑음, 기온 20°C, 습도 55%",
    }
    result = weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")
    print(f"[get_weather] {city}: {result}")
    return result


@tool
def search_database(query: str, limit: int = 5) -> str:
    """데이터베이스에서 정보를 검색합니다.

    Args:
        query: 검색할 키워드
        limit: 반환할 최대 결과 수 (기본값: 5)

    Returns:
        검색 결과 JSON 문자열
    """
    # 시뮬레이션 데이터
    mock_results = [
        {"id": i, "title": f"{query} 관련 항목 {i}", "score": 0.9 - i * 0.1}
        for i in range(1, min(limit + 1, 6))
    ]
    print(f"[search_database] '{query}' 검색 → {len(mock_results)}개 결과")
    return json.dumps(mock_results, ensure_ascii=False)


# ============================================================
# STEP 2: 도구 목록 구성 및 LLM 바인딩
# ============================================================
# 사용할 도구들을 리스트로 묶고, LLM에 바인딩합니다.
# bind_tools()는 LLM에게 "이 도구들을 사용할 수 있다"고 알려줍니다.

tools = [add_numbers, multiply_numbers, calculate_circle_area, get_weather, search_database]

# LLM 초기화
base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 도구를 바인딩한 LLM
# 이 LLM은 응답할 때 도구를 호출하는 AIMessage를 생성할 수 있습니다
llm_with_tools = base_llm.bind_tools(tools)


# ============================================================
# STEP 3: State 정의
# ============================================================

class ToolAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ============================================================
# STEP 4: Agent 노드 정의
# ============================================================

def agent_node(state: ToolAgentState) -> dict:
    """
    LLM을 호출해서 도구 사용 여부를 결정하는 노드.

    LLM은 두 가지 방식으로 응답합니다:
    1. 도구 호출이 필요한 경우:
       AIMessage(tool_calls=[{"name": "tool_name", "args": {...}}])
    2. 최종 답변인 경우:
       AIMessage(content="최종 답변 텍스트")
    """
    print(f"[agent_node] LLM 호출 (메시지 수: {len(state['messages'])})")
    response = llm_with_tools.invoke(state["messages"])

    # 도구 호출 여부 확인
    if response.tool_calls:
        print(f"[agent_node] 도구 호출 요청: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print(f"[agent_node] 최종 응답 생성")

    return {"messages": [response]}


# ============================================================
# STEP 5: 그래프 구성
# ============================================================
# tools_condition: 내장 라우터 함수
#   - 마지막 메시지가 tool_calls를 포함하면 → "tools" 노드로
#   - tool_calls가 없으면 → END로

def build_tool_agent():
    """
    도구 사용 에이전트 그래프.

    구조:
    START → agent → (도구 필요?) → tools → agent → ... → END
                                         ↑
                          (계속 반복하면서 도구 사용)
    """
    graph_builder = StateGraph(ToolAgentState)

    # 노드 추가
    graph_builder.add_node("agent", agent_node)

    # ToolNode: 도구 실행 전용 내장 노드
    # tool_calls에 있는 도구를 자동으로 찾아서 실행하고
    # 결과를 ToolMessage로 State에 추가합니다
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    # 엣지 추가
    graph_builder.add_edge(START, "agent")

    # tools_condition: 자동 라우팅
    # - tool_calls 있음 → "tools" 노드
    # - tool_calls 없음 → END
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,  # 내장 라우터 함수
    )

    # 도구 실행 후 다시 agent로 돌아감 (결과를 보고 다음 행동 결정)
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile()


# ============================================================
# 도구 메타데이터 확인
# ============================================================

def inspect_tools():
    """정의된 도구들의 메타데이터를 확인합니다"""
    print("=== 도구 목록 ===")
    for t in tools:
        print(f"\n도구명: {t.name}")
        print(f"설명: {t.description}")
        print(f"파라미터: {t.args_schema.schema()['properties']}")


# ============================================================
# 도구 직접 실행 테스트
# ============================================================

def test_tools_directly():
    """도구를 직접 호출해서 테스트합니다"""
    print("\n=== 도구 직접 실행 테스트 ===")

    # @tool 데코레이터가 붙은 함수는 .invoke()로 호출
    print(add_numbers.invoke({"a": 15, "b": 27}))
    print(calculate_circle_area.invoke({"radius": 5}))
    print(get_weather.invoke({"city": "서울"}))


# ============================================================
# 전체 에이전트 실행 테스트
# ============================================================

def run_agent_demo():
    """에이전트를 실행해서 도구 사용을 테스트합니다"""
    print("\n=== 에이전트 도구 사용 테스트 ===")
    agent = build_tool_agent()

    test_questions = [
        "서울 날씨가 어때요?",
        "반지름이 7인 원의 넓이를 계산해주세요",
        "15 더하기 27은 얼마예요?",
    ]

    for question in test_questions:
        print(f"\n질문: {question}")
        print("-" * 40)
        try:
            result = agent.invoke({
                "messages": [HumanMessage(content=question)]
            })
            # 최종 AI 응답 출력
            final_response = result["messages"][-1]
            print(f"최종 답변: {final_response.content}")
        except Exception as e:
            print(f"⚠️ API 오류 (API 키 필요): {e}")


# ============================================================
# 메시지 흐름 시각화
# ============================================================

def visualize_tool_call_flow():
    """도구 호출 시 메시지 흐름을 보여주는 예제"""
    print("\n=== 도구 호출 메시지 흐름 ===")
    print("""
    1. HumanMessage:
       content="서울 날씨가 어때요?"

    2. AIMessage (tool_calls 포함):
       content=""
       tool_calls=[{
           "name": "get_weather",
           "args": {"city": "서울"},
           "id": "call_abc123"
       }]

    3. ToolMessage (도구 실행 결과):
       content="맑음, 기온 22°C, 습도 60%"
       tool_call_id="call_abc123"

    4. AIMessage (최종 답변):
       content="서울의 현재 날씨는 맑음이며, 기온은 22°C입니다."
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph ToolNode 예제")
    print("=" * 60)

    inspect_tools()
    test_tools_directly()
    visualize_tool_call_flow()

    print("\n" + "=" * 60)
    print("에이전트 실행 (API 키 필요):")
    run_agent_demo()
