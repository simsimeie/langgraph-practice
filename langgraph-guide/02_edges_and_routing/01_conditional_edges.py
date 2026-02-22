"""
LangGraph 조건부 엣지 (Conditional Edges)
==========================================

목표: 상태에 따라 다른 경로로 분기하는 그래프를 만든다.

핵심 개념:
- add_conditional_edges(): 조건부 엣지 추가 메서드
- 라우터 함수: State를 받아서 다음 노드 이름을 반환하는 함수
- Literal 타입: 반환 가능한 값을 명시적으로 제한
- 루프(Loop): 조건이 충족될 때까지 같은 노드를 반복

사용 케이스:
- 입력 분류 후 전문 노드로 라우팅
- 품질 검사 후 통과/재시도 분기
- 에러 발생 시 복구 경로로 전환
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


# ============================================================
# 예제 1: 단순 분류 라우팅
# ============================================================
# 사용자 입력을 분류해서 적절한 처리 노드로 보내는 패턴

class ClassifyState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # 분류 결과를 저장하는 필드
    category: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def classify_node(state: ClassifyState) -> dict:
    """
    사용자 입력을 분류하는 노드.
    기술/일반/불명확 중 하나로 분류합니다.
    """
    last_message = state["messages"][-1].content

    # LLM으로 분류 (실제로는 더 정교한 프롬프트 필요)
    # 여기서는 간단히 키워드로 분류
    content_lower = last_message.lower()

    if any(kw in content_lower for kw in ["코드", "python", "에러", "버그", "함수", "프로그래밍"]):
        category = "technical"
    elif any(kw in content_lower for kw in ["안녕", "반가워", "좋은", "날씨"]):
        category = "casual"
    else:
        category = "general"

    print(f"[classify_node] 분류 결과: {category}")
    return {"category": category}


def technical_node(state: ClassifyState) -> dict:
    """기술적인 질문을 처리하는 전문 노드"""
    print("[technical_node] 기술 질문 처리 중")
    user_msg = state["messages"][-1].content
    # 실제로는 LLM 호출
    response = AIMessage(content=f"[기술 전문가] '{user_msg}'에 대한 기술적 답변입니다.")
    return {"messages": [response]}


def casual_node(state: ClassifyState) -> dict:
    """일상적인 대화를 처리하는 노드"""
    print("[casual_node] 일상 대화 처리 중")
    response = AIMessage(content="안녕하세요! 오늘 좋은 하루 보내세요! 😊")
    return {"messages": [response]}


def general_node(state: ClassifyState) -> dict:
    """일반 질문을 처리하는 노드"""
    print("[general_node] 일반 질문 처리 중")
    user_msg = state["messages"][-1].content
    response = AIMessage(content=f"'{user_msg}'에 대해 도움을 드리겠습니다.")
    return {"messages": [response]}


# ── 라우터 함수 ──────────────────────────────────────────────
# 라우터 함수는 State를 받아서 다음 노드의 이름(문자열)을 반환합니다.
# Literal 타입을 사용하면 반환 가능한 값이 명시적으로 제한됩니다.
# 이는 코드 가독성과 오류 방지에 도움이 됩니다.

def route_by_category(state: ClassifyState) -> Literal["technical", "casual", "general"]:
    """
    분류 결과에 따라 다음 노드를 결정하는 라우터 함수.

    Returns:
        다음으로 이동할 노드의 이름
    """
    category = state["category"]
    print(f"[라우터] '{category}' 노드로 이동")
    return category  # type: ignore


def build_classify_graph():
    """분류 기반 라우팅 그래프 구성"""
    graph_builder = StateGraph(ClassifyState)

    # 노드 추가
    graph_builder.add_node("classify", classify_node)
    graph_builder.add_node("technical", technical_node)
    graph_builder.add_node("casual", casual_node)
    graph_builder.add_node("general", general_node)

    # 일반 엣지
    graph_builder.add_edge(START, "classify")

    # ── 조건부 엣지 추가 ──────────────────────────────────────
    # add_conditional_edges(소스_노드, 라우터_함수, [경로_맵])
    #
    # 소스_노드: 이 노드가 실행된 후 조건부 라우팅이 적용됨
    # 라우터_함수: State를 받아 다음 노드 이름을 반환
    # 경로_맵 (선택): {"반환값": "노드이름"} 매핑 (필요한 경우)
    #
    # 라우터 함수가 반환하는 값이 노드 이름과 같으면 경로_맵 생략 가능
    graph_builder.add_conditional_edges(
        "classify",           # 이 노드 실행 후 분기
        route_by_category,    # 이 함수로 다음 노드 결정
        # 경로_맵 생략: 함수 반환값이 노드 이름과 동일하므로
    )

    # 모든 처리 노드 → END
    graph_builder.add_edge("technical", END)
    graph_builder.add_edge("casual", END)
    graph_builder.add_edge("general", END)

    return graph_builder.compile()


# ============================================================
# 예제 2: 루프를 활용한 품질 검사
# ============================================================
# 생성된 결과물이 기준을 통과할 때까지 반복하는 패턴
# 예: 글쓰기 → 검토 → 기준 미달이면 재작성 → 통과하면 완료

class QualityState(TypedDict):
    # 작성 중인 텍스트
    content: str
    # 품질 점수 (0-100)
    quality_score: int
    # 재시도 횟수 (무한루프 방지용)
    retry_count: int
    # 최대 재시도 횟수
    max_retries: int


def generate_content(state: QualityState) -> dict:
    """컨텐츠를 생성/개선하는 노드"""
    retry = state["retry_count"]

    if retry == 0:
        # 첫 번째 생성
        content = "이것은 초안 텍스트입니다. 아직 개선이 필요합니다."
        print(f"[generate] 초안 생성")
    else:
        # 재시도 시 개선
        content = f"이것은 {retry}번 개선된 텍스트입니다. 품질이 향상되었습니다."
        print(f"[generate] {retry}번째 개선")

    return {"content": content}


def evaluate_quality(state: QualityState) -> dict:
    """품질을 평가하는 노드"""
    retry = state["retry_count"]

    # 시뮬레이션: 재시도할수록 점수가 높아짐
    score = min(40 + retry * 25, 100)
    print(f"[evaluate] 품질 점수: {score}/100")

    return {
        "quality_score": score,
        "retry_count": retry + 1  # 재시도 횟수 증가
    }


def route_quality(state: QualityState) -> Literal["generate_content", "end"]:
    """
    품질 점수와 재시도 횟수를 보고 다음 동작 결정.

    Returns:
        "generate_content": 품질 부족 → 재생성
        "end": 품질 충족 또는 최대 재시도 → 종료
    """
    score = state["quality_score"]
    retries = state["retry_count"]
    max_retries = state["max_retries"]

    if score >= 80:
        print(f"[라우터] 품질 통과 (점수: {score}) → 완료")
        return "end"
    elif retries >= max_retries:
        print(f"[라우터] 최대 재시도 도달 ({retries}회) → 강제 완료")
        return "end"
    else:
        print(f"[라우터] 품질 부족 (점수: {score}) → 재생성")
        return "generate_content"


def build_quality_loop_graph():
    """품질 검사 루프 그래프"""
    graph_builder = StateGraph(QualityState)

    graph_builder.add_node("generate_content", generate_content)
    graph_builder.add_node("evaluate_quality", evaluate_quality)

    graph_builder.add_edge(START, "generate_content")
    graph_builder.add_edge("generate_content", "evaluate_quality")

    # 조건부 엣지: evaluate_quality → (generate_content 또는 END)
    # 경로_맵을 사용해서 문자열 "end"를 END 상수로 매핑
    graph_builder.add_conditional_edges(
        "evaluate_quality",
        route_quality,
        {
            "generate_content": "generate_content",  # 재생성 루프
            "end": END                                # 종료
        }
    )

    return graph_builder.compile()


# ============================================================
# 예제 3: 다중 조건 라우팅 (여러 분기점)
# ============================================================
# 하나의 그래프에서 여러 곳에 조건부 엣지가 있는 패턴

class MultiRouteState(TypedDict):
    input_text: str
    language: str        # 언어 감지 결과
    intent: str          # 의도 분류 결과
    result: str


def detect_language(state: MultiRouteState) -> dict:
    """언어 감지 노드"""
    text = state["input_text"]
    # 간단한 한국어 감지 (실제로는 LLM이나 langdetect 사용)
    has_korean = any('\uAC00' <= c <= '\uD7A3' for c in text)
    language = "korean" if has_korean else "english"
    print(f"[detect_language] 감지된 언어: {language}")
    return {"language": language}


def classify_intent(state: MultiRouteState) -> dict:
    """의도 분류 노드"""
    text = state["input_text"].lower()
    if "?" in state["input_text"] or "what" in text or "어떻게" in state["input_text"]:
        intent = "question"
    elif "please" in text or "해줘" in state["input_text"]:
        intent = "request"
    else:
        intent = "statement"
    print(f"[classify_intent] 감지된 의도: {intent}")
    return {"intent": intent}


def route_language(state: MultiRouteState) -> Literal["classify_intent_ko", "classify_intent_en"]:
    return "classify_intent_ko" if state["language"] == "korean" else "classify_intent_en"


def classify_intent_ko(state: MultiRouteState) -> dict:
    print("[classify_intent_ko] 한국어 의도 분류")
    return {"intent": "한국어 질문"}


def classify_intent_en(state: MultiRouteState) -> dict:
    print("[classify_intent_en] English intent classification")
    return {"intent": "English question"}


def respond_node(state: MultiRouteState) -> dict:
    return {"result": f"[{state['language']}/{state['intent']}] 처리 완료"}


def build_multi_route_graph():
    """다중 조건 라우팅 그래프"""
    graph_builder = StateGraph(MultiRouteState)

    graph_builder.add_node("detect_language", detect_language)
    graph_builder.add_node("classify_intent_ko", classify_intent_ko)
    graph_builder.add_node("classify_intent_en", classify_intent_en)
    graph_builder.add_node("respond", respond_node)

    graph_builder.add_edge(START, "detect_language")

    # 첫 번째 분기: 언어에 따라 라우팅
    graph_builder.add_conditional_edges(
        "detect_language",
        route_language
    )

    # 두 번째 분기 없음 - 두 언어 처리 노드 모두 respond로 이동
    graph_builder.add_edge("classify_intent_ko", "respond")
    graph_builder.add_edge("classify_intent_en", "respond")
    graph_builder.add_edge("respond", END)

    return graph_builder.compile()


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("조건부 엣지 예제")
    print("=" * 60)

    # 예제 1: 분류 라우팅
    print("\n[예제 1: 분류 기반 라우팅]")
    classify_graph = build_classify_graph()

    test_inputs = [
        "파이썬 코드에서 에러가 나요",
        "안녕하세요!",
        "오늘 날씨가 어때요?"
    ]

    for text in test_inputs:
        print(f"\n입력: '{text}'")
        result = classify_graph.invoke({
            "messages": [HumanMessage(content=text)],
            "category": ""
        })
        print(f"응답: {result['messages'][-1].content}")

    # 예제 2: 품질 검사 루프
    print("\n" + "-" * 60)
    print("[예제 2: 품질 검사 루프]")
    quality_graph = build_quality_loop_graph()
    result = quality_graph.invoke({
        "content": "",
        "quality_score": 0,
        "retry_count": 0,
        "max_retries": 5
    })
    print(f"최종 컨텐츠: {result['content']}")
    print(f"최종 점수: {result['quality_score']}")

    # 예제 3: 다중 라우팅
    print("\n" + "-" * 60)
    print("[예제 3: 다중 조건 라우팅]")
    multi_graph = build_multi_route_graph()

    for text in ["안녕하세요, 도움이 필요해요!", "Hello, how are you?"]:
        print(f"\n입력: '{text}'")
        result = multi_graph.invoke({
            "input_text": text,
            "language": "",
            "intent": "",
            "result": ""
        })
        print(f"결과: {result['result']}")
