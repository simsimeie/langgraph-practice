"""
LangGraph 간단한 챗봇 구현
============================

목표: LLM과 LangGraph를 연동해서 멀티턴 대화 챗봇을 만든다.

핵심 개념:
- ChatOpenAI: LLM 인스턴스 생성
- HumanMessage / AIMessage: 역할별 메시지 타입
- add_messages: 대화 기록을 자동으로 관리하는 Reducer
- graph.invoke() vs graph.stream(): 실행 방식의 차이

대화 흐름:
  사용자 입력 → [chatbot 노드] → LLM 응답 → 대화 기록에 추가 → 반복

채팅봇 설계 원칙:
  - State에 메시지 히스토리를 저장한다
  - 각 노드는 전체 히스토리를 LLM에게 전달한다
  - LLM은 히스토리를 보고 문맥을 이해하여 응답한다
"""

from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 환경변수 로드 (.env 파일에서 OPENAI_API_KEY 읽기)
load_dotenv()


# ============================================================
# State 정의
# ============================================================
# 챗봇의 대화 기록을 관리하는 State입니다.
# add_messages Reducer를 사용해서 새 메시지가 올 때마다
# 기존 기록에 이어붙이도록 합니다.

class ChatState(TypedDict):
    # 대화 기록 (add_messages Reducer 사용)
    # ✅ HumanMessage, AIMessage, SystemMessage 등이 모두 저장됩니다
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# LLM 초기화
# ============================================================
# ChatOpenAI: OpenAI의 채팅 모델을 사용합니다.
# model: 사용할 모델 이름 (gpt-4o-mini, gpt-4o, gpt-3.5-turbo 등)
# temperature: 창의성 조절 (0 = 결정적, 1 = 창의적)
# streaming: True로 설정하면 토큰별로 스트리밍 가능

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)


# ============================================================
# Chatbot Node
# ============================================================
# 가장 단순한 형태의 챗봇 노드입니다.
# 메시지 히스토리 전체를 LLM에 전달하고 응답을 받습니다.

def chatbot_node(state: ChatState) -> dict:
    """
    LLM을 호출해서 응답을 생성하는 노드.

    동작:
    1. state['messages']에서 전체 대화 기록을 가져옴
    2. LLM에 전달해서 응답 생성
    3. 새 AIMessage를 반환 (add_messages가 기록에 추가함)
    """
    # 전체 대화 기록을 LLM에 전달
    # LLM은 이전 대화를 모두 보고 문맥에 맞는 응답을 생성합니다
    response = llm.invoke(state["messages"])

    # add_messages Reducer가 자동으로 기존 기록에 추가해줍니다
    return {"messages": [response]}


# ============================================================
# 시스템 메시지 포함 챗봇 (더 나은 버전)
# ============================================================
# SystemMessage로 AI의 역할과 행동 방식을 설정합니다.
# 예: "당신은 친절한 한국어 도우미입니다."

SYSTEM_PROMPT = """당신은 친절하고 도움이 되는 AI 어시스턴트입니다.
한국어로 대화하며, 간결하고 명확하게 답변합니다.
모르는 것은 모른다고 솔직하게 말합니다."""

def chatbot_with_system(state: ChatState) -> dict:
    """
    시스템 프롬프트가 포함된 더 향상된 챗봇 노드.

    시스템 메시지를 매번 앞에 추가해서 AI의 행동을 일관되게 유지합니다.
    """
    # 시스템 메시지 + 기존 대화 기록을 합쳐서 전달
    messages_with_system = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"]  # 기존 메시지들을 펼쳐서 추가
    ]

    response = llm.invoke(messages_with_system)
    return {"messages": [response]}


# ============================================================
# 그래프 구성
# ============================================================

def build_simple_chatbot():
    """
    기본 챗봇 그래프.

    구조: START → chatbot → END

    단일 노드 챗봇 - 매우 단순한 구조입니다.
    """
    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()


def build_system_chatbot():
    """시스템 프롬프트가 있는 챗봇 그래프."""
    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("chatbot", chatbot_with_system)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()


# ============================================================
# 단일 질문 실행
# ============================================================

def single_query_demo(graph):
    """단일 질문-답변 예제"""
    print("\n=== 단일 질문 예제 ===")

    # HumanMessage로 사용자 입력 생성
    user_message = HumanMessage(content="파이썬의 리스트와 튜플의 차이점은?")

    # 그래프 실행
    result = graph.invoke({"messages": [user_message]})

    # 마지막 메시지가 AI 응답입니다
    ai_response = result["messages"][-1]
    print(f"Q: {user_message.content}")
    print(f"A: {ai_response.content}")


# ============================================================
# 멀티턴 대화 구현
# ============================================================
# LangGraph 자체는 대화 상태를 자동으로 유지하지 않습니다.
# 멀티턴 대화를 하려면 이전 결과를 다음 호출에 전달해야 합니다.
#
# 방법 1: 이전 state의 messages를 새 입력과 합쳐서 전달
# 방법 2: Checkpointer 사용 (04_memory 챕터에서 자세히 다룹니다)

def multi_turn_demo(graph):
    """수동으로 메시지 히스토리를 관리하는 멀티턴 대화"""
    print("\n=== 멀티턴 대화 예제 (수동 히스토리 관리) ===")

    # 대화 기록을 직접 관리합니다
    conversation_history = []

    questions = [
        "LangGraph가 뭔가요?",
        "그럼 LangChain과의 차이점은?",
        "어떤 상황에서 LangGraph를 사용하는 게 좋을까요?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n[턴 {i}]")
        print(f"사용자: {question}")

        # 새 질문을 히스토리에 추가
        conversation_history.append(HumanMessage(content=question))

        # 전체 히스토리를 포함해서 그래프 실행
        result = graph.invoke({"messages": conversation_history})

        # AI 응답 추출
        ai_message = result["messages"][-1]
        print(f"AI: {ai_message.content[:200]}...")  # 200자까지만 출력

        # 다음 턴을 위해 히스토리에 AI 응답 추가
        # 주의: result['messages']에는 전체 기록이 있으므로
        #       마지막 AI 메시지만 추가합니다
        conversation_history.append(ai_message)

    print(f"\n총 대화 메시지 수: {len(conversation_history)}")


# ============================================================
# 스트리밍 출력
# ============================================================
# stream()을 사용하면 LLM의 응답을 토큰 단위로 실시간 출력할 수 있습니다.

def streaming_demo(graph):
    """스트리밍으로 응답을 실시간 출력하는 예제"""
    print("\n=== 스트리밍 출력 예제 ===")
    print("AI의 응답이 생성되는 대로 실시간으로 출력됩니다:")
    print("AI: ", end="", flush=True)

    # stream()은 각 노드의 업데이트를 이벤트로 순서대로 반환합니다
    for event in graph.stream(
        {"messages": [HumanMessage(content="파이썬 비동기 프로그래밍에 대해 한 문장으로 설명해줘")]},
        stream_mode="values"  # 각 스텝의 전체 state를 반환
    ):
        # 마지막 메시지가 업데이트될 때 출력
        if "messages" in event:
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage):
                print(last_msg.content, end="", flush=True)

    print()  # 줄바꿈


# ============================================================
# 대화 기록 포맷팅
# ============================================================

def format_conversation(messages: list[BaseMessage]) -> str:
    """대화 기록을 읽기 좋은 형식으로 포맷팅"""
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"👤 사용자: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"🤖 AI: {msg.content}")
        elif isinstance(msg, SystemMessage):
            formatted.append(f"⚙️  시스템: {msg.content}")
    return "\n".join(formatted)


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 챗봇 예제")
    print("=" * 60)
    print("주의: OPENAI_API_KEY 환경변수가 필요합니다")
    print()

    # 시스템 프롬프트가 있는 챗봇 사용 (더 나은 버전)
    chatbot = build_system_chatbot()

    # 그래프 구조 출력
    print("그래프 구조:")
    print(chatbot.get_graph().draw_mermaid())

    # 각 예제 실행
    try:
        single_query_demo(chatbot)
        multi_turn_demo(chatbot)
        streaming_demo(chatbot)
    except Exception as e:
        print(f"\n⚠️  API 키가 없거나 오류 발생: {e}")
        print("실제 실행을 위해 .env 파일에 OPENAI_API_KEY를 설정하세요.")

    print("\n" + "=" * 60)
    print("학습 포인트:")
    print("1. ChatState에 add_messages Reducer를 사용합니다")
    print("2. 단일 노드도 완전한 그래프입니다")
    print("3. 멀티턴은 이전 messages를 다음 invoke에 포함시킵니다")
    print("4. Checkpointer로 자동 히스토리 관리가 가능합니다 (04장)")
    print("=" * 60)
