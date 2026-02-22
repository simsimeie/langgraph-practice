"""
LangGraph SQLite 체크포인터 - 영구 저장
========================================

목표: SQLite를 사용해서 대화 기록을 파일에 영구적으로 저장하고 재개한다.

핵심 개념:
- SqliteSaver: SQLite 파일에 State를 저장하는 체크포인터
- 영구 저장: 프로세스를 재시작해도 대화 기록이 유지됨
- 컨텍스트 매니저: with 문으로 안전하게 DB 연결 관리
- 비동기 버전: AsyncSqliteSaver (FastAPI 등에서 사용)

MemorySaver vs SqliteSaver:
  MemorySaver: 빠름, 임시 저장, 프로세스 종료 시 소멸
  SqliteSaver: 느림, 영구 저장, 프로세스 재시작 후에도 유지

실제 배포 시 선택 기준:
  - 개발/테스트: MemorySaver
  - 단일 서버 배포: SqliteSaver
  - 클러스터/분산: PostgresSaver (langgraph-checkpoint-postgres 패키지)
"""

import os
from typing import Annotated, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# SQLite 체크포인터 임포트
# 설치: pip install langgraph-checkpoint-sqlite
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    SQLITE_AVAILABLE = True
except ImportError:
    print("⚠️ langgraph-checkpoint-sqlite 미설치")
    print("설치: pip install langgraph-checkpoint-sqlite")
    SQLITE_AVAILABLE = False
    # 폴백: MemorySaver 사용
    from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# SQLite 파일 경로 설정
DB_PATH = "./chat_history.db"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ============================================================
# State 및 노드 정의
# ============================================================

class PersistentChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_name: str      # 사용자 이름 기억


SYSTEM_TEMPLATE = """당신은 친절한 AI 어시스턴트입니다.
사용자 이름: {user_name}
이전 대화를 기억하고 personalized된 응답을 제공하세요."""


def chatbot_node(state: PersistentChatState) -> dict:
    """персонализированный 챗봇 노드"""
    user_name = state.get("user_name", "사용자")
    system_msg = SystemMessage(content=SYSTEM_TEMPLATE.format(user_name=user_name))
    messages = [system_msg, *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


def build_graph():
    """그래프 빌더만 반환 (체크포인터는 나중에 추가)"""
    graph_builder = StateGraph(PersistentChatState)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder


# ============================================================
# SqliteSaver 사용법
# ============================================================

def run_with_sqlite():
    """
    SQLite 체크포인터를 사용한 영구 저장 예제.

    컨텍스트 매니저(with 문)를 사용하는 것을 권장합니다.
    이렇게 하면 with 블록이 끝날 때 DB 연결이 자동으로 닫힙니다.
    """
    if not SQLITE_AVAILABLE:
        print("SqliteSaver를 사용할 수 없습니다. MemorySaver를 사용합니다.")
        run_with_memory_fallback()
        return

    print("=== SQLite 체크포인터 예제 ===")
    print(f"데이터베이스 파일: {DB_PATH}")

    # ── 컨텍스트 매니저로 안전한 DB 연결 ──────────────────────
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        # 그래프 빌드 및 컴파일 (체크포인터 포함)
        graph = build_graph().compile(checkpointer=checkpointer)

        # 사용자 설정
        user_config = {
            "configurable": {
                "thread_id": "persistent_user_001"
            }
        }

        # 첫 번째 세션: 초기 대화
        print("\n[첫 번째 세션 시작]")
        initial_state = {
            "messages": [HumanMessage(content="안녕하세요! 저는 김철수입니다.")],
            "user_name": "김철수"
        }

        try:
            result = graph.invoke(initial_state, config=user_config)
            print(f"AI: {result['messages'][-1].content[:200]}")

            # 두 번째 메시지
            result2 = graph.invoke(
                {"messages": [HumanMessage(content="제 이름이 뭔가요?")],
                 "user_name": "김철수"},
                config=user_config
            )
            print(f"\n사용자: 제 이름이 뭔가요?")
            print(f"AI: {result2['messages'][-1].content[:200]}")

        except Exception as e:
            print(f"⚠️ LLM 오류: {e}")

    print(f"\n✅ DB 파일 저장됨: {os.path.abspath(DB_PATH)}")


def simulate_session_restart():
    """
    프로세스 재시작 후 이전 대화 재개 시뮬레이션.

    실제 애플리케이션에서 유용한 패턴:
    - 사용자가 앱을 재시작해도 이전 대화가 유지됨
    - 서버 재시작 후에도 세션 복원 가능
    """
    if not SQLITE_AVAILABLE:
        return

    print("\n=== 세션 재시작 시뮬레이션 ===")

    # 1단계: 초기 대화 생성 (첫 번째 실행)
    print("\n[1단계: 초기 대화 생성]")
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "restart_demo"}}

        try:
            graph.invoke(
                {"messages": [HumanMessage(content="내 좋아하는 색은 파란색이야")],
                 "user_name": "테스트유저"},
                config=config
            )
            print("대화 저장 완료")
        except Exception as e:
            print(f"⚠️ LLM 오류: {e}")

    # 2단계: 새 연결로 이전 대화 재개 (재시작 시뮬레이션)
    print("\n[2단계: 재시작 후 이전 대화 재개]")
    with SqliteSaver.from_conn_string(DB_PATH) as new_checkpointer:
        # 새 연결이지만 같은 DB 파일 사용
        new_graph = build_graph().compile(checkpointer=new_checkpointer)
        same_config = {"configurable": {"thread_id": "restart_demo"}}

        # 이전 대화 확인
        state = new_graph.get_state(same_config)
        if state.values:
            msg_count = len(state.values.get('messages', []))
            print(f"재로드된 메시지 수: {msg_count}")

        try:
            result = new_graph.invoke(
                {"messages": [HumanMessage(content="내가 좋아하는 색이 뭔지 알아?")],
                 "user_name": "테스트유저"},
                config=same_config
            )
            print(f"AI: {result['messages'][-1].content[:200]}")
        except Exception as e:
            print(f"⚠️ LLM 오류: {e}")


def list_all_threads():
    """저장된 모든 스레드 목록 조회"""
    if not SQLITE_AVAILABLE or not os.path.exists(DB_PATH):
        print("DB 파일이 없습니다.")
        return

    print("\n=== 저장된 대화 스레드 목록 ===")
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph().compile(checkpointer=checkpointer)

        # 알려진 thread_id들의 상태 확인
        test_thread_ids = ["persistent_user_001", "restart_demo", "time_travel_demo"]

        for thread_id in test_thread_ids:
            config = {"configurable": {"thread_id": thread_id}}
            state = graph.get_state(config)
            if state.values:
                msg_count = len(state.values.get('messages', []))
                print(f"  Thread '{thread_id}': 메시지 {msg_count}개")


def run_with_memory_fallback():
    """SqliteSaver를 사용할 수 없을 때 MemorySaver로 폴백"""
    from langgraph.checkpoint.memory import MemorySaver

    print("\n=== MemorySaver 폴백 예제 ===")
    checkpointer = MemorySaver()
    graph = build_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "fallback_test"}}

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content="폴백 테스트")],
             "user_name": "테스터"},
            config=config
        )
        print(f"AI: {result['messages'][-1].content[:100]}")
    except Exception as e:
        print(f"⚠️ LLM 오류 (API 키 필요): {e}")


# ============================================================
# 체크포인트 정보 구조 이해하기
# ============================================================

def explain_checkpoint_structure():
    """
    체크포인트의 내부 구조를 설명합니다.
    실제 체크포인트 데이터가 어떻게 생겼는지 이해하는 데 도움이 됩니다.
    """
    print("\n=== 체크포인트 구조 설명 ===")
    print("""
체크포인트 스냅샷 구조:
{
    "v": 1,                          # 버전
    "id": "uuid-...",                # 체크포인트 고유 ID
    "ts": "2024-01-01T00:00:00Z",   # 타임스탬프
    "channel_values": {              # 현재 State 값
        "messages": [...],           # 메시지 목록
        "user_name": "김철수"
    },
    "channel_versions": {           # 각 채널의 버전
        "messages": 5,
        "user_name": 2
    },
    "versions_seen": {...},         # 이전에 본 버전들
    "pending_sends": []             # 대기 중인 이벤트
}

config 구조:
{
    "configurable": {
        "thread_id": "user_001",        # 대화 세션 ID
        "checkpoint_ns": "",            # 네임스페이스 (서브그래프용)
        "checkpoint_id": "uuid-..."     # 특정 체크포인트로 이동 시 사용
    }
}
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph SQLite 체크포인터 예제")
    print("=" * 60)

    explain_checkpoint_structure()
    run_with_sqlite()
    simulate_session_restart()
    list_all_threads()

    # 실행 후 DB 파일 정리 (선택사항)
    if os.path.exists(DB_PATH):
        print(f"\n생성된 파일: {os.path.abspath(DB_PATH)}")
        print("(파일을 삭제하면 모든 대화 기록이 사라집니다)")
