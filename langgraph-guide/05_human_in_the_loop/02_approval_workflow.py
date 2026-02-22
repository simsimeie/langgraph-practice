"""
LangGraph 실전 승인 워크플로우
================================

목표: 이메일 발송, 코드 실행 등 중요 작업에 사람의 승인을 요구하는 실전 패턴을 구현한다.

시나리오: AI 코드 실행 에이전트
- AI가 코드를 작성
- 사람이 코드를 검토하고 승인/거절/수정
- 승인된 경우에만 실행

이 패턴은 다음 상황에서 유용합니다:
- 파일 삭제/수정 작업
- 이메일/메시지 발송
- 결제/거래 처리
- 코드 자동 배포
- 데이터베이스 변경
"""

from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# ============================================================
# State 정의
# ============================================================

class CodeAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # 생성된 코드
    generated_code: str
    # 코드 검토 결과
    review_feedback: str
    # 최종 실행 결과
    execution_result: str
    # 실행 승인 여부
    is_approved: bool


# ============================================================
# 도구 정의
# ============================================================

@tool
def write_to_file(filename: str, content: str) -> str:
    """파일에 내용을 작성합니다. 중요한 작업이므로 사람의 검토가 필요합니다.

    Args:
        filename: 파일 이름
        content: 파일에 쓸 내용
    """
    # 시뮬레이션 (실제로는 파일에 씁니다)
    print(f"[write_to_file] 파일: {filename}")
    return f"파일 '{filename}'에 {len(content)}자 작성 완료"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """이메일을 발송합니다. 중요한 작업이므로 사람의 검토가 필요합니다.

    Args:
        to: 수신자 이메일
        subject: 이메일 제목
        body: 이메일 본문
    """
    # 시뮬레이션
    print(f"[send_email] To: {to}, Subject: {subject}")
    return f"이메일이 {to}에게 발송되었습니다."


# ============================================================
# 코드 에이전트 - 도구 사용 에이전트
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [write_to_file, send_email]
llm_with_tools = llm.bind_tools(tools)


def planning_node(state: CodeAgentState) -> dict:
    """사용자 요청을 분석하고 실행 계획을 수립하는 노드"""
    print("[planning] 요청 분석 중...")
    system = SystemMessage(content="""당신은 파이썬 코드를 작성하는 AI 어시스턴트입니다.
사용자의 요청을 분석하고 필요한 코드를 작성하세요.
도구가 필요한 경우 적절히 사용하세요.""")

    messages = [system, *state["messages"]]
    response = llm.invoke(messages)
    print(f"[planning] 계획 수립 완료")
    return {"messages": [response]}


def code_generation_node(state: CodeAgentState) -> dict:
    """Python 코드를 생성하는 노드"""
    print("[code_generation] 코드 생성 중...")

    last_msg = state["messages"][-1].content
    # 시뮬레이션: 코드 생성
    generated_code = f"""# 자동 생성된 코드
# 요청: {last_msg[:50]}

import os
import json

def main():
    data = {{"status": "success", "message": "작업 완료"}}
    with open("output.json", "w") as f:
        json.dump(data, f)
    print("작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
"""
    print(f"[code_generation] {len(generated_code)}자 코드 생성")
    return {
        "generated_code": generated_code,
        "messages": [AIMessage(content=f"코드를 생성했습니다:\n```python\n{generated_code}\n```")]
    }


def human_code_review_node(state: CodeAgentState) -> dict:
    """
    사람이 생성된 코드를 검토하는 노드.

    interrupt()로 실행을 중단하고 사람의 피드백을 기다립니다.
    사람은 다음 중 하나를 선택할 수 있습니다:
    - "approve": 코드를 그대로 실행
    - "reject": 코드 실행 취소
    - "modify:수정사항": 코드를 수정 후 실행
    """
    print("[human_review] 코드 검토 대기 중...")

    # 사람에게 코드 검토 요청
    # interrupt()의 반환값이 human_input이 됩니다
    human_input = interrupt({
        "type": "code_review",
        "message": "다음 코드를 검토해주세요:",
        "code": state["generated_code"],
        "options": {
            "approve": "코드를 승인하고 실행합니다",
            "reject": "코드 실행을 취소합니다",
            "modify:<피드백>": "코드를 수정합니다. 예: 'modify:파일명을 output.txt로 변경해주세요'"
        }
    })

    print(f"[human_review] 사람의 입력: {human_input}")

    # 입력 파싱
    if human_input.startswith("modify:"):
        feedback = human_input[7:]  # "modify:" 제거
        return {
            "review_feedback": feedback,
            "is_approved": False
        }
    elif human_input.lower() == "approve":
        return {
            "review_feedback": "승인됨",
            "is_approved": True
        }
    else:
        return {
            "review_feedback": "거절됨",
            "is_approved": False
        }


def apply_feedback_node(state: CodeAgentState) -> dict:
    """사람의 피드백을 반영해서 코드를 수정하는 노드"""
    feedback = state["review_feedback"]
    original_code = state["generated_code"]
    print(f"[apply_feedback] 피드백 반영: {feedback}")

    # 실제로는 LLM으로 코드 수정
    modified_code = f"""# 수정된 코드 (피드백: {feedback})
{original_code}
# 위 코드가 수정되었습니다
"""
    return {
        "generated_code": modified_code,
        "messages": [AIMessage(content=f"피드백을 반영해서 코드를 수정했습니다: {feedback}")]
    }


def execute_code_node(state: CodeAgentState) -> dict:
    """승인된 코드를 실행하는 노드"""
    if not state["is_approved"]:
        print("[execute_code] 코드가 승인되지 않았습니다.")
        return {"execution_result": "실행 취소됨"}

    print("[execute_code] 코드 실행 중...")
    # 실제로는 코드를 실행합니다 (예: exec(), subprocess 등)
    # 여기서는 시뮬레이션
    result = "코드 실행 성공! output.json 파일이 생성되었습니다."
    return {
        "execution_result": result,
        "messages": [AIMessage(content=f"코드 실행 완료: {result}")]
    }


# ============================================================
# 라우팅 함수
# ============================================================

def route_after_review(state: CodeAgentState):
    """검토 결과에 따른 라우팅"""
    feedback = state["review_feedback"]

    if feedback == "승인됨":
        print("[라우터] 승인 → 코드 실행")
        return "execute_code"
    elif feedback == "거절됨":
        print("[라우터] 거절 → 종료")
        return "__end__"
    else:
        # "modify:..." 피드백이 있는 경우
        print("[라우터] 수정 요청 → 코드 재생성")
        return "apply_feedback"


# ============================================================
# 그래프 구성
# ============================================================

def build_code_approval_graph():
    """
    코드 생성 → 검토 → (승인/수정/거절) 워크플로우 그래프.

    START
      │
      ▼
    [planning]
      │
      ▼
    [code_generation]
      │
      ▼
    [human_review] ← interrupt() 발생
      │
      ├─ 승인 ────────────────▶ [execute_code] → END
      ├─ 수정 ▶ [apply_feedback] → [human_review] (다시 검토)
      └─ 거절 ────────────────────────────────▶ END
    """
    checkpointer = MemorySaver()
    graph_builder = StateGraph(CodeAgentState)

    graph_builder.add_node("planning", planning_node)
    graph_builder.add_node("code_generation", code_generation_node)
    graph_builder.add_node("human_review", human_code_review_node)
    graph_builder.add_node("apply_feedback", apply_feedback_node)
    graph_builder.add_node("execute_code", execute_code_node)

    graph_builder.add_edge(START, "planning")
    graph_builder.add_edge("planning", "code_generation")
    graph_builder.add_edge("code_generation", "human_review")

    # 검토 후 분기
    graph_builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "execute_code": "execute_code",
            "apply_feedback": "apply_feedback",
            "__end__": END
        }
    )

    # 수정 후 다시 검토로
    graph_builder.add_edge("apply_feedback", "human_review")
    graph_builder.add_edge("execute_code", END)

    return graph_builder.compile(checkpointer=checkpointer)


# ============================================================
# 시뮬레이션 실행
# ============================================================

def simulate_approve_workflow():
    """승인 시나리오 시뮬레이션"""
    print("\n=== 코드 승인 워크플로우 시뮬레이션 ===")
    graph = build_code_approval_graph()
    config = {"configurable": {"thread_id": "code_review_001"}}

    initial_state = {
        "messages": [HumanMessage(content="결과를 JSON 파일로 저장하는 코드를 작성해줘")],
        "generated_code": "",
        "review_feedback": "",
        "execution_result": "",
        "is_approved": False
    }

    print("\n[1단계] 그래프 실행 시작")
    try:
        graph.invoke(initial_state, config=config)
    except Exception as e:
        # interrupt 발생
        print(f"\n[interrupt 발생] 코드 검토 필요")
        print("(실제 앱에서는 여기서 사용자에게 UI를 표시합니다)")

    # 현재 상태 확인
    state = graph.get_state(config)
    print(f"\n현재 State:")
    print(f"  다음 노드: {state.next}")
    print(f"  생성된 코드 길이: {len(state.values.get('generated_code', ''))}자")

    # interrupt 내용 확인
    if state.tasks:
        for task in state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                interrupt_data = task.interrupts[0].value
                print(f"\n[검토 요청 내용]")
                print(f"  메시지: {interrupt_data.get('message', '')}")
                options = interrupt_data.get('options', {})
                for key, desc in options.items():
                    print(f"  - {key}: {desc}")

    print("\n[2단계] 코드 승인")
    try:
        final = graph.invoke(Command(resume="approve"), config=config)
        print(f"\n최종 실행 결과: {final.get('execution_result', '없음')}")
    except Exception as e:
        print(f"오류: {e}")


def simulate_modify_workflow():
    """수정 요청 시나리오 시뮬레이션"""
    print("\n=== 코드 수정 워크플로우 시뮬레이션 ===")
    graph = build_code_approval_graph()
    config = {"configurable": {"thread_id": "code_review_002"}}

    # 1단계: 실행 시작
    try:
        graph.invoke({
            "messages": [HumanMessage(content="파일 저장 코드 작성")],
            "generated_code": "",
            "review_feedback": "",
            "execution_result": "",
            "is_approved": False
        }, config=config)
    except Exception:
        pass

    print("[수정 요청] '파일명을 result.json으로 변경해줘'")

    # 2단계: 수정 요청
    try:
        graph.invoke(
            Command(resume="modify:파일명을 result.json으로 변경해줘"),
            config=config
        )
    except Exception:
        # 수정 후 다시 검토 interrupt 발생
        print("[다시 검토 필요] 수정된 코드를 검토합니다")

    # 3단계: 수정된 코드 승인
    print("[최종 승인]")
    try:
        final = graph.invoke(Command(resume="approve"), config=config)
        print(f"최종 결과: {final.get('execution_result', '없음')}")
    except Exception as e:
        print(f"오류: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Human-in-the-Loop 승인 워크플로우")
    print("=" * 60)

    simulate_approve_workflow()
    simulate_modify_workflow()

    print("\n" + "=" * 60)
    print("핵심 패턴 정리:")
    print("1. 중요 노드에 interrupt() 배치")
    print("2. 사람이 확인 후 Command(resume=...)로 재개")
    print("3. 수정이 필요하면 루프로 돌아와서 재검토")
    print("4. 거절하면 그래프 종료")
    print("=" * 60)
