"""
LangGraph 고급 라우팅 패턴
============================

목표: 실전에서 자주 쓰이는 라우팅 패턴들을 이해한다.

패턴 목록:
1. 체인 오브 씽킹 (CoT) 라우팅: 단계적 추론 후 경로 결정
2. 폴백(Fallback) 패턴: 기본 경로 실패 시 대체 경로
3. 팬아웃/팬인: 하나에서 여럿으로, 여럿에서 하나로 합류
4. 게이트키퍼 패턴: 입력 검증 후 통과/거절
"""

from typing import Annotated, Literal, List
import operator
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages


# ============================================================
# 패턴 1: LLM 기반 라우팅 (실전 패턴)
# ============================================================
# LLM을 사용해서 사용자 의도를 파악하고 적절한 노드로 라우팅합니다.
# 실제 프로덕션에서 가장 많이 사용되는 패턴입니다.

class IntentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # LLM이 결정한 다음 행동
    next_action: str


def llm_router_node(state: IntentState) -> dict:
    """
    LLM을 사용해서 다음 행동을 결정하는 라우터 노드.

    실제 구현에서는 Structured Output이나 함수 호출로
    더 안정적인 라우팅이 가능합니다.
    """
    # 실제로는 아래처럼 LLM으로 의도 파악:
    # response = llm.invoke([
    #     SystemMessage("다음 중 하나로만 답하세요: search, calculate, translate, chat"),
    #     *state["messages"]
    # ])
    # next_action = response.content.strip().lower()

    # 시뮬레이션용 키워드 기반 라우팅
    last_msg = state["messages"][-1].content.lower()

    if any(kw in last_msg for kw in ["검색", "찾아", "알려줘", "search"]):
        next_action = "search"
    elif any(kw in last_msg for kw in ["계산", "더하기", "곱하기", "calculate"]):
        next_action = "calculate"
    elif any(kw in last_msg for kw in ["번역", "translate", "영어로", "한국어로"]):
        next_action = "translate"
    else:
        next_action = "chat"

    print(f"[llm_router] 결정된 액션: {next_action}")
    return {"next_action": next_action}


def search_node(state: IntentState) -> dict:
    print("[search_node] 검색 실행")
    return {"messages": [AIMessage(content="검색 결과: ... (검색 기능 시뮬레이션)")]}


def calculate_node(state: IntentState) -> dict:
    print("[calculate_node] 계산 실행")
    return {"messages": [AIMessage(content="계산 결과: 42 (계산 기능 시뮬레이션)")]}


def translate_node(state: IntentState) -> dict:
    print("[translate_node] 번역 실행")
    return {"messages": [AIMessage(content="Translation: Hello! (번역 기능 시뮬레이션)")]}


def chat_node(state: IntentState) -> dict:
    print("[chat_node] 일반 대화")
    return {"messages": [AIMessage(content="안녕하세요! 무엇을 도와드릴까요?")]}


def route_by_action(state: IntentState) -> Literal["search", "calculate", "translate", "chat"]:
    """라우터 노드가 결정한 next_action을 그대로 반환"""
    return state["next_action"]  # type: ignore


def build_intent_router():
    graph = StateGraph(IntentState)
    graph.add_node("router", llm_router_node)
    graph.add_node("search", search_node)
    graph.add_node("calculate", calculate_node)
    graph.add_node("translate", translate_node)
    graph.add_node("chat", chat_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_by_action)

    # 모든 처리 노드 → END
    for node in ["search", "calculate", "translate", "chat"]:
        graph.add_edge(node, END)

    return graph.compile()


# ============================================================
# 패턴 2: 폴백(Fallback) 패턴
# ============================================================
# 기본 처리가 실패하면 대체 경로로 이동합니다.
# 에러 처리, 재시도 로직에 유용합니다.

class FallbackState(TypedDict):
    input_data: str
    result: str
    error: str       # 에러 발생 시 에러 메시지
    attempts: int


def primary_processor(state: FallbackState) -> dict:
    """기본 처리 노드 - 실패할 수 있음"""
    print(f"[primary] 처리 시도 #{state['attempts'] + 1}")

    # 시뮬레이션: 첫 번째 시도는 실패
    if state["attempts"] == 0:
        return {
            "error": "처리 실패: 데이터 형식 오류",
            "attempts": state["attempts"] + 1
        }
    else:
        return {
            "result": f"성공적으로 처리됨: {state['input_data']}",
            "error": "",
            "attempts": state["attempts"] + 1
        }


def fallback_processor(state: FallbackState) -> dict:
    """폴백 처리 노드 - 더 안전한 방식으로 처리"""
    print(f"[fallback] 폴백 처리 실행")
    return {
        "result": f"폴백으로 처리됨: {state['input_data']} (단순 처리)",
        "error": ""
    }


def route_with_fallback(state: FallbackState) -> Literal["primary_processor", "fallback_processor", "end"]:
    """에러 여부에 따라 폴백 또는 완료 결정"""
    if state["error"]:
        if state["attempts"] < 2:
            print(f"[라우터] 에러 발생 → 재시도")
            return "primary_processor"
        else:
            print(f"[라우터] 재시도 한계 → 폴백으로 전환")
            return "fallback_processor"
    else:
        print(f"[라우터] 처리 성공 → 완료")
        return "end"


def build_fallback_graph():
    graph = StateGraph(FallbackState)
    graph.add_node("primary_processor", primary_processor)
    graph.add_node("fallback_processor", fallback_processor)

    graph.add_edge(START, "primary_processor")
    graph.add_conditional_edges(
        "primary_processor",
        route_with_fallback,
        {
            "primary_processor": "primary_processor",
            "fallback_processor": "fallback_processor",
            "end": END
        }
    )
    graph.add_edge("fallback_processor", END)

    return graph.compile()


# ============================================================
# 패턴 3: 게이트키퍼(Gatekeeper) 패턴
# ============================================================
# 입력을 먼저 검증하고, 유효한 경우만 처리 파이프라인으로 보냅니다.
# 보안, 입력 검증에 유용합니다.

class GatekeeperState(TypedDict):
    user_input: str
    is_valid: bool
    validation_errors: Annotated[List[str], operator.add]
    result: str


def validate_input(state: GatekeeperState) -> dict:
    """입력 유효성 검사 노드"""
    text = state["user_input"]
    errors = []

    if not text or not text.strip():
        errors.append("입력이 비어있습니다")
    if len(text) > 500:
        errors.append("입력이 너무 깁니다 (최대 500자)")
    if any(bad_word in text.lower() for bad_word in ["욕설1", "욕설2"]):
        errors.append("부적절한 내용이 포함되어 있습니다")

    is_valid = len(errors) == 0
    print(f"[validate] 유효성: {is_valid}, 에러: {errors}")

    return {
        "is_valid": is_valid,
        "validation_errors": errors
    }


def process_valid_input(state: GatekeeperState) -> dict:
    """유효한 입력을 처리하는 노드"""
    print("[process] 유효한 입력 처리 중")
    return {"result": f"처리 완료: '{state['user_input']}'"}


def reject_invalid_input(state: GatekeeperState) -> dict:
    """유효하지 않은 입력을 거절하는 노드"""
    error_msg = ", ".join(state["validation_errors"])
    print(f"[reject] 입력 거절: {error_msg}")
    return {"result": f"입력 오류: {error_msg}"}


def route_gatekeeper(state: GatekeeperState) -> Literal["process_valid_input", "reject_invalid_input"]:
    return "process_valid_input" if state["is_valid"] else "reject_invalid_input"


def build_gatekeeper_graph():
    graph = StateGraph(GatekeeperState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("process_valid_input", process_valid_input)
    graph.add_node("reject_invalid_input", reject_invalid_input)

    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges("validate_input", route_gatekeeper)
    graph.add_edge("process_valid_input", END)
    graph.add_edge("reject_invalid_input", END)

    return graph.compile()


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("고급 라우팅 패턴 예제")
    print("=" * 60)

    # 패턴 1: 의도 기반 라우터
    print("\n[패턴 1: LLM 기반 의도 라우팅]")
    intent_graph = build_intent_router()

    for question in ["파이썬에 대해 검색해줘", "100 곱하기 42 계산해줘", "안녕하세요"]:
        print(f"\n질문: '{question}'")
        result = intent_graph.invoke({
            "messages": [HumanMessage(content=question)],
            "next_action": ""
        })
        print(f"응답: {result['messages'][-1].content}")

    # 패턴 2: 폴백
    print("\n" + "-" * 60)
    print("[패턴 2: 폴백 패턴]")
    fallback_graph = build_fallback_graph()
    result = fallback_graph.invoke({
        "input_data": "테스트 데이터",
        "result": "",
        "error": "",
        "attempts": 0
    })
    print(f"최종 결과: {result['result']}")

    # 패턴 3: 게이트키퍼
    print("\n" + "-" * 60)
    print("[패턴 3: 게이트키퍼]")
    gate_graph = build_gatekeeper_graph()

    for test_input in ["안녕하세요, 도움을 요청합니다!", "", "a" * 600]:
        label = "정상" if test_input and len(test_input) < 500 else "비정상"
        print(f"\n입력({label}): '{test_input[:50]}...' " if len(test_input) > 50 else f"\n입력({label}): '{test_input}'")
        result = gate_graph.invoke({
            "user_input": test_input,
            "is_valid": False,
            "validation_errors": [],
            "result": ""
        })
        print(f"결과: {result['result']}")
