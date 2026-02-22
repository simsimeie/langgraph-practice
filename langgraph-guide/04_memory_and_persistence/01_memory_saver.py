"""
LangGraph MemorySaver - 인메모리 체크포인터
=============================================

목표: MemorySaver를 사용해서 대화 기록이 자동으로 유지되는 챗봇을 만든다.

핵심 개념:
- MemorySaver: 메모리에 State를 저장하는 가장 단순한 체크포인터
- thread_id: 대화 세션을 구분하는 식별자
- config: 그래프 실행 시 설정을 전달하는 딕셔너리
- get_state(): 저장된 State 조회
- get_state_history(): State 변경 이력 조회

MemorySaver의 특징:
- 메모리에만 저장 → 프로세스 종료 시 모든 데이터 소멸
- 설정이 간단하고 빠름
- 개발/테스트 환경에 적합
- 프로덕션에서는 SQLite나 PostgreSQL 사용 권장
"""

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
# State 정의
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ============================================================
# LLM 및 노드 정의
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

SYSTEM_PROMPT = """당신은 친절한 AI 대화 상대입니다.
이전 대화 내용을 기억하고 문맥에 맞게 대화합니다."""


def chatbot_node(state: ChatState) -> dict:
    """챗봇 노드"""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ============================================================
# 체크포인터를 포함한 그래프 구성
# ============================================================

def build_persistent_chatbot():
    """
    MemorySaver 체크포인터를 사용하는 챗봇 그래프.

    compile(checkpointer=checkpointer)를 추가하면
    모든 노드 실행 후 State가 자동으로 저장됩니다.
    """
    # 체크포인터 생성
    # MemorySaver: 딕셔너리 기반 인메모리 저장
    checkpointer = MemorySaver()

    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # 중요: checkpointer를 compile()에 전달
    # 이 설정 하나로 자동 상태 저장이 활성화됩니다
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph, checkpointer


# ============================================================
# Thread ID를 활용한 멀티턴 대화
# ============================================================

def multi_turn_with_checkpointer():
    """
    체크포인터를 사용한 멀티턴 대화 예제.

    thread_id를 같게 유지하면 이전 대화가 자동으로 기억됩니다.
    """
    print("=== MemorySaver 멀티턴 대화 예제 ===\n")
    graph, checkpointer = build_persistent_chatbot()

    # ── 사용자 A의 대화 스레드 ────────────────────────────────
    # 모든 메시지에 같은 thread_id를 사용하면 대화가 이어집니다
    thread_a_config = {"configurable": {"thread_id": "user_alice"}}

    print("[사용자 Alice의 대화]")
    print("-" * 40)

    conversations = [
        "안녕하세요! 제 이름은 Alice입니다.",
        "제 이름을 기억하고 있나요?",
        "저는 파이썬 개발자예요.",
        "제 직업이 뭔지 알고 있나요?",
    ]

    for user_msg in conversations:
        print(f"\n사용자: {user_msg}")

        # 각 invoke 호출 시 동일한 thread_id 사용
        # LangGraph가 자동으로 이전 메시지들을 State에서 불러옵니다
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config=thread_a_config  # 중요: config에 thread_id 포함
        )

        ai_response = result["messages"][-1].content
        print(f"AI: {ai_response[:200]}")


def multi_thread_demo():
    """
    여러 독립적인 대화 스레드를 보여주는 예제.

    다른 thread_id를 사용하면 완전히 독립된 대화가 시작됩니다.
    """
    print("\n=== 멀티 스레드 독립 대화 예제 ===\n")
    graph, _ = build_persistent_chatbot()

    users = [
        ("thread_alice", "안녕! 나는 Alice야. 파이썬을 좋아해."),
        ("thread_bob", "Hi! I'm Bob. I love JavaScript."),
        ("thread_alice", "내가 좋아하는 프로그래밍 언어가 뭔지 기억해?"),
        ("thread_bob", "Do you remember what programming language I like?"),
    ]

    for thread_id, message in users:
        config = {"configurable": {"thread_id": thread_id}}
        print(f"\n[{thread_id}] 메시지: {message}")
        result = graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        ai_msg = result["messages"][-1].content
        print(f"[{thread_id}] AI: {ai_msg[:150]}")


# ============================================================
# 저장된 State 조회
# ============================================================

def inspect_saved_state():
    """
    저장된 State를 조회하는 방법들.

    체크포인터를 사용하면 State를 언제든지 읽을 수 있습니다.
    """
    print("\n=== 저장된 State 조회 ===\n")
    graph, checkpointer = build_persistent_chatbot()

    config = {"configurable": {"thread_id": "inspect_demo"}}

    # 몇 가지 메시지를 보냄
    messages = ["LangGraph에 대해 설명해줘", "더 자세히 알려줘"]
    for msg in messages:
        graph.invoke(
            {"messages": [HumanMessage(content=msg)]},
            config=config
        )

    # ── get_state(): 현재 State 조회 ──────────────────────────
    current_state = graph.get_state(config)
    print(f"현재 State 조회:")
    print(f"  메시지 수: {len(current_state.values.get('messages', []))}")
    print(f"  다음 노드: {current_state.next}")  # [] = 실행 완료

    # ── get_state_history(): 전체 실행 이력 ───────────────────
    print(f"\n실행 이력:")
    for i, state_snapshot in enumerate(graph.get_state_history(config)):
        msg_count = len(state_snapshot.values.get('messages', []))
        step = state_snapshot.metadata.get('step', '?')
        print(f"  스냅샷 #{i}: 메시지 수={msg_count}, 단계={step}")


# ============================================================
# 특정 시점으로 되돌리기 (Time Travel)
# ============================================================

def time_travel_demo():
    """
    특정 체크포인트로 되돌아가는 Time Travel 기능.

    그래프 실행의 특정 시점으로 되돌아가서
    다시 실행하거나 상태를 수정할 수 있습니다.

    활용 사례:
    - 디버깅: 어떤 단계에서 잘못됐는지 확인
    - 실험: 다른 입력으로 시뮬레이션
    - 수정: 잘못된 응답을 수정 후 재실행
    """
    print("\n=== Time Travel (상태 되돌리기) 예제 ===")
    print("주의: thread_id와 checkpoint_id를 사용합니다\n")
    graph, checkpointer = build_persistent_chatbot()

    config = {"configurable": {"thread_id": "time_travel_demo"}}

    # 3번의 대화 진행
    for i, msg in enumerate(["첫 번째 메시지", "두 번째 메시지", "세 번째 메시지"]):
        graph.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
        print(f"메시지 {i+1} 전송 완료")

    # 실행 이력에서 checkpoint_id 수집
    history = list(graph.get_state_history(config))
    print(f"\n총 {len(history)}개 스냅샷 저장됨")

    if len(history) >= 2:
        # 두 번째로 오래된 스냅샷으로 되돌아가기
        target_checkpoint = history[-2]
        checkpoint_id = target_checkpoint.config["configurable"]["checkpoint_id"]

        print(f"\n체크포인트 {checkpoint_id[:20]}...으로 되돌아가기")
        print(f"해당 시점의 메시지 수: {len(target_checkpoint.values.get('messages', []))}")

        # 특정 checkpoint_id로 State 로드
        # 이 config로 invoke하면 해당 시점부터 다시 실행됩니다
        past_config = {
            "configurable": {
                "thread_id": "time_travel_demo",
                "checkpoint_id": checkpoint_id
            }
        }

        past_state = graph.get_state(past_config)
        print(f"되돌아간 시점의 메시지 수: {len(past_state.values.get('messages', []))}")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph MemorySaver 예제")
    print("=" * 60)

    # 각 예제를 순서대로 실행 (API 키 필요)
    try:
        multi_turn_with_checkpointer()
        multi_thread_demo()
        inspect_saved_state()
        time_travel_demo()
    except Exception as e:
        print(f"\n⚠️ API 오류 (API 키 필요): {e}")

    print("\n" + "=" * 60)
    print("학습 포인트:")
    print("1. compile(checkpointer=MemorySaver())로 자동 저장 활성화")
    print("2. 동일한 thread_id = 대화가 이어짐")
    print("3. 다른 thread_id = 새 독립 대화 시작")
    print("4. get_state() / get_state_history()로 저장 상태 조회")
    print("5. checkpoint_id로 특정 시점으로 되돌리기 가능")
    print("=" * 60)
