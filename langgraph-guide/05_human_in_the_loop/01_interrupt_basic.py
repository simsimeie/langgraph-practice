"""
LangGraph interrupt() 기본 사용법
===================================

목표: interrupt()를 사용해서 그래프 실행을 일시 중단하고 재개하는 방법을 이해한다.

핵심 개념:
- interrupt(value): 실행을 일시 중단하고 value를 반환
- Command(resume=value): 중단된 실행을 재개하는 명령
- GraphInterrupt 예외: interrupt() 호출 시 발생하는 예외

interrupt()의 작동 방식:
1. 노드 안에서 interrupt() 호출
2. 그래프 실행이 일시 중단됨 (GraphInterrupt 예외 발생)
3. 사람이 상황 검토
4. Command(resume=...)로 재개
5. interrupt()가 resume 값을 반환하고 노드 계속 실행

필수 조건: 반드시 checkpointer가 있어야 합니다!
"""

from typing import Annotated, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command  # 핵심 임포트

load_dotenv()


# ============================================================
# State 정의
# ============================================================

class InterruptState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    action_requested: str   # 요청된 작업
    action_approved: bool   # 승인 여부
    result: str             # 작업 결과


# ============================================================
# 노드 정의
# ============================================================

def prepare_action_node(state: InterruptState) -> dict:
    """실행할 작업을 준비하는 노드"""
    print("[prepare_action] 작업 준비 중...")
    action = "데이터베이스에서 100개의 레코드 삭제"
    print(f"[prepare_action] 요청된 작업: {action}")
    return {"action_requested": action}


def human_review_node(state: InterruptState) -> dict:
    """
    사람의 검토를 요청하는 노드.

    interrupt()를 호출하면:
    1. 그래프 실행이 여기서 멈춤
    2. interrupt()에 전달한 값이 그래프 호출자에게 반환됨
    3. Command(resume=...)로 재개하면 interrupt()가 resume 값을 반환함

    interrupt()의 인자: 사람에게 보여줄 정보 (딕셔너리, 문자열 등 아무 값)
    interrupt()의 반환값: Command(resume=...)에서 전달한 값
    """
    print("[human_review] 사람의 검토 대기 중...")

    # 사람에게 보여줄 정보를 담아서 interrupt() 호출
    # 실행이 여기서 일시 중단됩니다
    human_decision = interrupt({
        "question": "다음 작업을 승인하시겠습니까?",
        "action": state["action_requested"],
        "impact": "이 작업은 되돌릴 수 없습니다.",
        "instructions": "'approve' 또는 'reject'를 입력하세요"
    })

    # 사람이 Command(resume="approve")로 재개하면
    # human_decision = "approve"가 됩니다

    print(f"[human_review] 사람의 결정: {human_decision}")

    approved = human_decision.lower() == "approve"
    return {"action_approved": approved}


def execute_action_node(state: InterruptState) -> dict:
    """승인된 경우 실제 작업을 실행하는 노드"""
    if state["action_approved"]:
        print(f"[execute_action] ✅ 승인됨. 작업 실행: {state['action_requested']}")
        result = f"작업 완료: {state['action_requested']}"
    else:
        print(f"[execute_action] ❌ 거절됨. 작업 취소")
        result = "작업 취소됨"

    return {"result": result}


# ============================================================
# 그래프 구성
# ============================================================

def build_interrupt_graph():
    """interrupt를 포함한 그래프 구성"""
    # ⚠️ interrupt()를 사용하려면 checkpointer가 반드시 필요합니다!
    checkpointer = MemorySaver()

    graph_builder = StateGraph(InterruptState)
    graph_builder.add_node("prepare_action", prepare_action_node)
    graph_builder.add_node("human_review", human_review_node)
    graph_builder.add_node("execute_action", execute_action_node)

    graph_builder.add_edge(START, "prepare_action")
    graph_builder.add_edge("prepare_action", "human_review")
    graph_builder.add_edge("human_review", "execute_action")
    graph_builder.add_edge("execute_action", END)

    # checkpointer 없으면 interrupt가 작동하지 않습니다!
    return graph_builder.compile(checkpointer=checkpointer)


# ============================================================
# 실행: 승인 흐름
# ============================================================

def run_approval_flow():
    """
    사람이 승인하는 흐름:
    1. 그래프 실행 → interrupt에서 중단
    2. 사람이 내용 확인
    3. Command(resume="approve")로 재개
    4. 작업 실행
    """
    print("\n=== 승인 흐름 예제 ===")
    graph = build_interrupt_graph()
    config = {"configurable": {"thread_id": "approval_demo"}}

    # ── 1단계: 그래프 실행 시작 ────────────────────────────────
    print("\n[1단계] 그래프 실행 시작")
    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content="레코드 삭제 작업을 시작해주세요")],
                "action_requested": "",
                "action_approved": False,
                "result": ""
            },
            config=config
        )
        # interrupt()가 없으면 여기까지 오지만,
        # interrupt()가 있으면 예외가 발생해서 아래 except로 이동
        print(f"결과: {result}")

    except Exception as e:
        # interrupt()가 호출되면 GraphInterrupt 예외가 발생합니다
        # e.value에 interrupt()에 전달한 값이 들어있습니다
        interrupt_value = e.args[0] if e.args else str(e)
        print(f"\n[2단계] 실행 중단됨! 사람의 검토 필요:")
        print(f"  질문: {interrupt_value}")

    # ── 2단계: 사람이 상태 확인 ────────────────────────────────
    print("\n[2단계] 현재 그래프 상태 확인")
    current_state = graph.get_state(config)
    print(f"  다음 실행할 노드: {current_state.next}")  # ('human_review',)
    # interrupt 정보 확인
    if current_state.tasks:
        for task in current_state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                print(f"  Interrupt 내용: {task.interrupts[0].value}")

    # ── 3단계: 사람이 승인 결정 후 재개 ──────────────────────────
    print("\n[3단계] 승인 결정: 승인")
    try:
        # Command(resume=값)으로 interrupt()가 있는 노드를 재개
        # resume에 전달한 값이 interrupt()의 반환값이 됩니다
        final_result = graph.invoke(
            Command(resume="approve"),  # 사람의 결정을 전달
            config=config
        )
        print(f"\n최종 결과: {final_result.get('result', '알 수 없음')}")

    except Exception as e:
        print(f"오류: {e}")


# ============================================================
# 실행: 거절 흐름
# ============================================================

def run_rejection_flow():
    """사람이 거절하는 흐름"""
    print("\n=== 거절 흐름 예제 ===")
    graph = build_interrupt_graph()
    config = {"configurable": {"thread_id": "rejection_demo"}}

    # 실행 시작
    try:
        graph.invoke(
            {
                "messages": [HumanMessage(content="레코드 삭제 요청")],
                "action_requested": "",
                "action_approved": False,
                "result": ""
            },
            config=config
        )
    except Exception:
        pass  # interrupt 예외 무시

    # 거절로 재개
    try:
        final_result = graph.invoke(
            Command(resume="reject"),
            config=config
        )
        print(f"최종 결과: {final_result.get('result', '알 수 없음')}")
    except Exception as e:
        print(f"오류: {e}")


# ============================================================
# 실행: 수정(Update) 포함 재개
# ============================================================

def run_with_update_demo():
    """
    Command(resume=..., update=...)를 사용해서
    재개하면서 State를 수정하는 예제.

    사람이 검토 중에 State의 일부 값을 변경할 수 있습니다.
    예: 작업의 범위를 수정하거나 파라미터를 변경
    """
    print("\n=== update와 함께 재개 예제 ===")
    graph = build_interrupt_graph()
    config = {"configurable": {"thread_id": "update_demo"}}

    # 실행 시작
    try:
        graph.invoke(
            {
                "messages": [HumanMessage(content="작업 시작")],
                "action_requested": "",
                "action_approved": False,
                "result": ""
            },
            config=config
        )
    except Exception:
        pass

    # 재개하면서 State도 수정
    try:
        final_result = graph.invoke(
            Command(
                resume="approve",
                # update: 재개하면서 State의 특정 필드를 변경
                # 예: action_requested를 더 안전한 버전으로 수정
                update={
                    "action_requested": "데이터베이스에서 10개의 레코드만 삭제 (수정됨)"
                }
            ),
            config=config
        )
        print(f"최종 결과: {final_result.get('result', '알 수 없음')}")
    except Exception as e:
        print(f"오류: {e}")


# ============================================================
# interrupt() vs MemorySaver 없이 실행 시 오류 시연
# ============================================================

def demo_without_checkpointer():
    """
    체크포인터 없이 interrupt()를 사용하면 오류가 발생합니다.
    이 예제는 왜 체크포인터가 필요한지 보여줍니다.
    """
    print("\n=== 체크포인터 없이 interrupt() 사용 시 ===")

    def node_with_interrupt(state):
        interrupt("이 값은 저장되지 않습니다")  # checkpointer 없으면 오류
        return {}

    graph_builder = StateGraph(InterruptState)
    graph_builder.add_node("node", node_with_interrupt)
    graph_builder.add_edge(START, "node")
    graph_builder.add_edge("node", END)

    # 체크포인터 없이 컴파일
    graph_no_checkpoint = graph_builder.compile()

    try:
        graph_no_checkpoint.invoke({
            "messages": [], "action_requested": "",
            "action_approved": False, "result": ""
        })
    except ValueError as e:
        print(f"예상된 오류: {e}")
        print("→ interrupt()를 사용하려면 반드시 checkpointer가 필요합니다!")


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph interrupt() 기본 예제")
    print("=" * 60)

    run_approval_flow()
    run_rejection_flow()
    run_with_update_demo()
    demo_without_checkpointer()

    print("\n" + "=" * 60)
    print("학습 포인트:")
    print("1. interrupt(value)로 실행을 일시 중단")
    print("2. Command(resume=값)으로 재개")
    print("3. Command(resume=값, update={...})로 State 수정 후 재개")
    print("4. 반드시 checkpointer가 필요합니다!")
    print("=" * 60)
