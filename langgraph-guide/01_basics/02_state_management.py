"""
LangGraph State 관리 심화
==========================

목표: State 설계의 다양한 패턴과 Reducer 함수를 이해한다.

핵심 개념:
- Annotated: 필드에 메타데이터(Reducer)를 붙이는 Python 문법
- Reducer: 동일 키에 새 값이 들어올 때 어떻게 병합할지 정의하는 함수
- add_messages: 메시지 리스트를 올바르게 병합하는 내장 Reducer
- operator.add: 리스트를 이어붙이는 내장 Reducer

왜 Reducer가 필요한가?
  여러 노드가 같은 State 키를 업데이트하면 충돌이 생길 수 있습니다.
  Reducer는 "새 값과 기존 값을 어떻게 합칠지"를 정의합니다.
"""

import operator
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# ============================================================
# 패턴 1: 기본 State (Reducer 없음 = 덮어쓰기)
# ============================================================
# Reducer를 지정하지 않으면 새 값이 기존 값을 완전히 덮어씁니다.
# 단순한 값(숫자, 문자열 등)에 적합합니다.

class BasicState(TypedDict):
    name: str       # 새 값이 오면 완전히 덮어씀
    score: int      # 새 값이 오면 완전히 덮어씀
    active: bool    # 새 값이 오면 완전히 덮어씀


def basic_reducer_demo():
    """기본 덮어쓰기 동작 시연"""
    print("\n[패턴 1: 기본 덮어쓰기]")

    def node_a(state: BasicState) -> dict:
        # score를 업데이트 (name과 active는 그대로 유지됨)
        return {"score": state["score"] + 10}

    def node_b(state: BasicState) -> dict:
        # name을 변경
        return {"name": f"{state['name']}_updated"}

    graph = StateGraph(BasicState)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)

    result = graph.compile().invoke({
        "name": "Alice",
        "score": 100,
        "active": True
    })
    print(f"결과: {result}")
    # 출력: {'name': 'Alice_updated', 'score': 110, 'active': True}


# ============================================================
# 패턴 2: operator.add Reducer (리스트 이어붙이기)
# ============================================================
# Annotated[List[str], operator.add] 의미:
#   - 이 필드는 List[str] 타입이다
#   - 새 값이 오면 operator.add (리스트 덧셈 = 이어붙이기)를 사용한다
#
# 예: 기존 ["a", "b"] + 새 ["c"] = ["a", "b", "c"]

class ListState(TypedDict):
    # operator.add를 Reducer로 사용하면 리스트가 이어붙여집니다
    items: Annotated[List[str], operator.add]
    # 일반 정수 필드 (덮어쓰기)
    total: int


def list_reducer_demo():
    """리스트 이어붙이기 Reducer 시연"""
    print("\n[패턴 2: operator.add Reducer]")

    def add_fruits(state: ListState) -> dict:
        # 기존 items에 새 과일들이 이어붙여집니다
        return {"items": ["사과", "바나나"]}

    def add_veggies(state: ListState) -> dict:
        # 기존 ["사과", "바나나"]에 ["당근", "브로콜리"]가 추가됩니다
        return {"items": ["당근", "브로콜리"]}

    def set_total(state: ListState) -> dict:
        return {"total": len(state["items"])}

    graph = StateGraph(ListState)
    graph.add_node("add_fruits", add_fruits)
    graph.add_node("add_veggies", add_veggies)
    graph.add_node("set_total", set_total)
    graph.add_edge(START, "add_fruits")
    graph.add_edge("add_fruits", "add_veggies")
    graph.add_edge("add_veggies", "set_total")
    graph.add_edge("set_total", END)

    result = graph.compile().invoke({"items": [], "total": 0})
    print(f"items: {result['items']}")
    print(f"total: {result['total']}")
    # 출력: items: ['사과', '바나나', '당근', '브로콜리'], total: 4


# ============================================================
# 패턴 3: add_messages Reducer (메시지 관리)
# ============================================================
# add_messages는 LangChain 메시지 리스트를 위한 특수 Reducer입니다.
#
# 일반 operator.add와의 차이점:
#   - 메시지의 id를 기반으로 중복을 자동으로 처리합니다
#   - 같은 id의 메시지가 오면 덮어씁니다 (메시지 수정 가능)
#   - 없는 메시지는 추가합니다
#
# 채팅 히스토리를 관리할 때 반드시 사용해야 합니다.

class ChatState(TypedDict):
    # add_messages Reducer로 메시지 리스트 관리
    messages: Annotated[List[BaseMessage], add_messages]
    # 일반 필드
    turn_count: int


def add_messages_demo():
    """add_messages Reducer 시연"""
    print("\n[패턴 3: add_messages Reducer]")

    def user_turn(state: ChatState) -> dict:
        # 새 HumanMessage를 추가 (기존 메시지들은 유지됨)
        new_message = HumanMessage(content="안녕하세요!")
        return {
            "messages": [new_message],  # add_messages가 기존 목록에 추가함
            "turn_count": state["turn_count"] + 1
        }

    def assistant_turn(state: ChatState) -> dict:
        # AI 응답 메시지 추가
        last_human_msg = state["messages"][-1].content
        ai_response = AIMessage(content=f"'{last_human_msg}'에 대한 응답: 반갑습니다!")
        return {"messages": [ai_response]}

    graph = StateGraph(ChatState)
    graph.add_node("user_turn", user_turn)
    graph.add_node("assistant_turn", assistant_turn)
    graph.add_edge(START, "user_turn")
    graph.add_edge("user_turn", "assistant_turn")
    graph.add_edge("assistant_turn", END)

    result = graph.compile().invoke({
        "messages": [],  # 빈 메시지 리스트로 시작
        "turn_count": 0
    })

    print(f"총 메시지 수: {len(result['messages'])}")
    for msg in result['messages']:
        role = "사용자" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{role}]: {msg.content}")
    print(f"턴 수: {result['turn_count']}")


# ============================================================
# 패턴 4: 커스텀 Reducer 함수
# ============================================================
# 직접 Reducer 함수를 만들어서 복잡한 병합 로직을 구현할 수 있습니다.

def score_reducer(existing: int, new: int) -> int:
    """
    점수를 단순 덮어쓰기가 아닌 누적 방식으로 합산합니다.
    기존 점수 + 새 점수 = 최종 점수
    """
    return existing + new


def set_reducer(existing: set, new: set) -> set:
    """
    집합(set)을 합집합으로 병합합니다.
    중복 없이 유니크한 값들만 유지합니다.
    """
    return existing | new  # 합집합


class CustomReducerState(TypedDict):
    # 커스텀 Reducer: 점수 누적
    score: Annotated[int, score_reducer]
    # 커스텀 Reducer: 집합 합집합
    visited_nodes: Annotated[set, set_reducer]
    # 일반 필드
    current_node: str


def custom_reducer_demo():
    """커스텀 Reducer 시연"""
    print("\n[패턴 4: 커스텀 Reducer]")

    def node_alpha(state: CustomReducerState) -> dict:
        return {
            "score": 50,                    # 기존 점수 + 50
            "visited_nodes": {"alpha"},      # 집합에 "alpha" 추가
            "current_node": "alpha"
        }

    def node_beta(state: CustomReducerState) -> dict:
        return {
            "score": 30,                    # 기존 점수 + 30
            "visited_nodes": {"beta"},       # 집합에 "beta" 추가
            "current_node": "beta"
        }

    graph = StateGraph(CustomReducerState)
    graph.add_node("node_alpha", node_alpha)
    graph.add_node("node_beta", node_beta)
    graph.add_edge(START, "node_alpha")
    graph.add_edge("node_alpha", "node_beta")
    graph.add_edge("node_beta", END)

    result = graph.compile().invoke({
        "score": 0,
        "visited_nodes": set(),
        "current_node": "start"
    })

    print(f"최종 점수: {result['score']}")         # 0 + 50 + 30 = 80
    print(f"방문한 노드: {result['visited_nodes']}")  # {'alpha', 'beta'}
    print(f"현재 노드: {result['current_node']}")   # 'beta' (덮어쓰기)


# ============================================================
# 패턴 5: 복잡한 State 구조 (중첩 TypedDict)
# ============================================================
# 실제 애플리케이션에서는 더 복잡한 State가 필요합니다.

class UserProfile(TypedDict):
    name: str
    age: int
    preferences: List[str]


class AppState(TypedDict):
    # 사용자 프로필 (중첩 TypedDict)
    user: UserProfile
    # 채팅 기록
    messages: Annotated[List[BaseMessage], add_messages]
    # 에러 메시지 리스트 (이어붙이기)
    errors: Annotated[List[str], operator.add]
    # 현재 단계
    current_step: str


def complex_state_demo():
    """복잡한 State 구조 시연"""
    print("\n[패턴 5: 복잡한 State 구조]")

    def step_one(state: AppState) -> dict:
        user = state["user"]
        return {
            "messages": [AIMessage(content=f"{user['name']}님 환영합니다!")],
            "current_step": "greeting"
        }

    def step_two(state: AppState) -> dict:
        # 에러가 없으면 정상 처리
        if state["user"]["age"] < 0:
            return {"errors": ["나이는 양수여야 합니다"]}

        return {
            "messages": [AIMessage(content="나이 검증 완료!")],
            "current_step": "validation_complete"
        }

    graph = StateGraph(AppState)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", END)

    initial = {
        "user": {"name": "김철수", "age": 25, "preferences": ["AI", "Python"]},
        "messages": [],
        "errors": [],
        "current_step": "init"
    }

    result = graph.compile().invoke(initial)
    print(f"최종 단계: {result['current_step']}")
    print(f"메시지 수: {len(result['messages'])}")
    for msg in result['messages']:
        print(f"  AI: {msg.content}")
    print(f"에러: {result['errors']}")


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph State 관리 패턴 데모")
    print("=" * 60)

    basic_reducer_demo()
    list_reducer_demo()
    add_messages_demo()
    custom_reducer_demo()
    complex_state_demo()

    print("\n" + "=" * 60)
    print("학습 포인트 요약:")
    print("1. Reducer 없음 → 새 값으로 덮어쓰기")
    print("2. operator.add → 리스트 이어붙이기")
    print("3. add_messages → 메시지 ID 기반 스마트 병합")
    print("4. 커스텀 함수 → 원하는 방식으로 병합")
    print("=" * 60)
