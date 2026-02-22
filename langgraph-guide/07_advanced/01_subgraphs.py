"""
LangGraph 서브그래프 (Subgraph)
================================

목표: 복잡한 워크플로우를 작은 서브그래프로 분리하고 재사용하는 방법을 이해한다.

서브그래프를 사용하는 이유:
1. 모듈화: 복잡한 로직을 작은 단위로 분리
2. 재사용성: 같은 서브그래프를 여러 곳에서 사용
3. 테스트 용이성: 서브그래프 단독으로 테스트 가능
4. 팀 협업: 각 팀이 독립적인 서브그래프 개발

서브그래프의 특징:
- 서브그래프는 독립적인 State를 가질 수 있습니다
- 메인 그래프와 서브그래프는 State의 공통 키를 통해 통신합니다
- 서브그래프 내부는 메인 그래프에서 보이지 않습니다 (캡슐화)
"""

from typing import Annotated, List, TypedDict
import operator
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 패턴 1: 간단한 서브그래프
# ============================================================
# 서브그래프는 그냥 컴파일된 StateGraph입니다.
# 메인 그래프에서 일반 노드처럼 사용할 수 있습니다.

# ── 서브그래프 1: 텍스트 분석 ─────────────────────────────────

class AnalysisSubState(TypedDict):
    """서브그래프 전용 State"""
    text: str            # 분석할 텍스트 (메인과 공유)
    word_count: int      # 단어 수 (서브그래프 내부 결과)
    sentiment: str       # 감성 (서브그래프 내부 결과)
    analysis_result: str # 최종 분석 결과 (메인과 공유)


def count_words_node(state: AnalysisSubState) -> dict:
    """단어 수를 세는 노드"""
    count = len(state["text"].split())
    print(f"  [서브그래프/count_words] 단어 수: {count}")
    return {"word_count": count}


def analyze_sentiment_node(state: AnalysisSubState) -> dict:
    """감성을 분석하는 노드"""
    text_lower = state["text"].lower()
    if any(w in text_lower for w in ["좋아", "훌륭", "최고", "great", "good"]):
        sentiment = "긍정"
    elif any(w in text_lower for w in ["싫어", "나쁜", "최악", "bad", "terrible"]):
        sentiment = "부정"
    else:
        sentiment = "중립"
    print(f"  [서브그래프/analyze_sentiment] 감성: {sentiment}")
    return {"sentiment": sentiment}


def summarize_analysis_node(state: AnalysisSubState) -> dict:
    """분석 결과를 요약하는 노드"""
    result = f"단어 수: {state['word_count']}, 감성: {state['sentiment']}"
    print(f"  [서브그래프/summarize] {result}")
    return {"analysis_result": result}


# 서브그래프 빌드 및 컴파일
def build_analysis_subgraph():
    """텍스트 분석 서브그래프"""
    builder = StateGraph(AnalysisSubState)
    builder.add_node("count_words", count_words_node)
    builder.add_node("analyze_sentiment", analyze_sentiment_node)
    builder.add_node("summarize", summarize_analysis_node)

    builder.add_edge(START, "count_words")
    builder.add_edge("count_words", "analyze_sentiment")
    builder.add_edge("analyze_sentiment", "summarize")
    builder.add_edge("summarize", END)

    # 서브그래프도 compile() 필요
    return builder.compile()


# ── 메인 그래프 ───────────────────────────────────────────────

class MainState(TypedDict):
    """메인 그래프 State"""
    messages: Annotated[List[BaseMessage], add_messages]
    # 서브그래프와 공유하는 필드 (같은 이름이어야 함)
    text: str
    analysis_result: str
    final_response: str


def extract_text_node(state: MainState) -> dict:
    """사용자 메시지에서 분석할 텍스트를 추출"""
    last_msg = state["messages"][-1].content
    print(f"[메인/extract_text] 텍스트 추출: '{last_msg[:50]}...'")
    return {"text": last_msg}


def format_response_node(state: MainState) -> dict:
    """분석 결과를 최종 응답으로 포맷"""
    analysis = state["analysis_result"]
    response = f"텍스트 분석 완료!\n분석 결과: {analysis}"
    print(f"[메인/format_response] 응답 생성")
    return {
        "final_response": response,
        "messages": [AIMessage(content=response)]
    }


def build_main_graph_with_subgraph():
    """
    서브그래프를 포함한 메인 그래프.

    메인 그래프에서 서브그래프를 add_node()로 추가합니다.
    서브그래프가 실행될 때 State의 공통 키(text, analysis_result)가
    자동으로 공유됩니다.
    """
    # 서브그래프 인스턴스 생성
    analysis_subgraph = build_analysis_subgraph()

    builder = StateGraph(MainState)

    # 일반 노드 추가
    builder.add_node("extract_text", extract_text_node)

    # 서브그래프를 노드로 추가
    # 서브그래프의 State와 메인 State에 공통 키(text, analysis_result)가 있어야 함
    builder.add_node("text_analysis", analysis_subgraph)

    builder.add_node("format_response", format_response_node)

    builder.add_edge(START, "extract_text")
    builder.add_edge("extract_text", "text_analysis")
    builder.add_edge("text_analysis", "format_response")
    builder.add_edge("format_response", END)

    return builder.compile(checkpointer=MemorySaver())


# ============================================================
# 패턴 2: 서브그래프 재사용
# ============================================================
# 같은 서브그래프를 여러 곳에서 사용하는 패턴

class TranslationSubState(TypedDict):
    """번역 서브그래프 State"""
    source_text: str    # 원본 텍스트
    target_language: str  # 번역 대상 언어
    translated_text: str  # 번역된 텍스트


def translate_node(state: TranslationSubState) -> dict:
    """번역 노드"""
    print(f"  [서브그래프/translate] {state['target_language']}으로 번역 중...")
    # 시뮬레이션 (실제로는 번역 API나 LLM 사용)
    translations = {
        "영어": f"[EN] {state['source_text']}",
        "일본어": f"[JA] {state['source_text']}",
        "중국어": f"[ZH] {state['source_text']}",
    }
    translated = translations.get(state["target_language"], f"[?] {state['source_text']}")
    return {"translated_text": translated}


def build_translation_subgraph():
    """번역 서브그래프 (재사용 가능)"""
    builder = StateGraph(TranslationSubState)
    builder.add_node("translate", translate_node)
    builder.add_edge(START, "translate")
    builder.add_edge("translate", END)
    return builder.compile()


class MultiTranslationState(TypedDict):
    """다국어 번역 메인 State"""
    source_text: str
    target_language: str
    translated_text: str
    translations: Annotated[List[str], operator.add]


def set_english_target(state: MultiTranslationState) -> dict:
    return {"target_language": "영어"}


def set_japanese_target(state: MultiTranslationState) -> dict:
    return {"target_language": "일본어"}


def collect_translations(state: MultiTranslationState) -> dict:
    """번역 결과 수집 (실제로는 병렬 처리가 더 효율적)"""
    print(f"[collect] 번역 결과: {state.get('translations', [])}")
    return {}


# ============================================================
# 패턴 3: 중첩 서브그래프
# ============================================================
# 서브그래프 안에 또 다른 서브그래프가 있는 패턴

class InnerSubState(TypedDict):
    data: str
    processed: str


class OuterSubState(TypedDict):
    data: str
    processed: str
    validated: bool


def inner_process_node(state: InnerSubState) -> dict:
    """내부 서브그래프의 처리 노드"""
    print(f"    [내부 서브그래프/process] '{state['data']}' 처리")
    return {"processed": f"processed({state['data']})"}


def build_inner_subgraph():
    """내부(가장 깊은) 서브그래프"""
    builder = StateGraph(InnerSubState)
    builder.add_node("process", inner_process_node)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    return builder.compile()


def outer_validate_node(state: OuterSubState) -> dict:
    """외부 서브그래프의 검증 노드"""
    is_valid = len(state.get("processed", "")) > 0
    print(f"  [외부 서브그래프/validate] 유효성: {is_valid}")
    return {"validated": is_valid}


def build_outer_subgraph():
    """외부 서브그래프 (내부 서브그래프 포함)"""
    inner = build_inner_subgraph()
    builder = StateGraph(OuterSubState)
    builder.add_node("inner_processing", inner)  # 내부 서브그래프를 노드로
    builder.add_node("validate", outer_validate_node)
    builder.add_edge(START, "inner_processing")
    builder.add_edge("inner_processing", "validate")
    builder.add_edge("validate", END)
    return builder.compile()


class DeepMainState(TypedDict):
    data: str
    processed: str
    validated: bool
    messages: Annotated[List[BaseMessage], add_messages]


def build_nested_graph():
    """중첩 서브그래프를 포함한 메인 그래프"""
    outer = build_outer_subgraph()
    builder = StateGraph(DeepMainState)
    builder.add_node("nested_processing", outer)  # 외부 서브그래프를 노드로
    builder.add_edge(START, "nested_processing")
    builder.add_edge("nested_processing", END)
    return builder.compile()


# ============================================================
# 실행
# ============================================================

def run_subgraph_demo():
    """서브그래프 예제 실행"""
    print("\n[예제 1: 기본 서브그래프]")
    main_graph = build_main_graph_with_subgraph()

    result = main_graph.invoke({
        "messages": [HumanMessage(content="LangGraph는 정말 좋은 프레임워크입니다!")],
        "text": "",
        "analysis_result": "",
        "final_response": ""
    }, config={"configurable": {"thread_id": "subgraph_demo"}})

    print(f"\n최종 응답: {result['final_response']}")


def run_nested_graph_demo():
    """중첩 서브그래프 예제 실행"""
    print("\n[예제 2: 중첩 서브그래프]")
    nested = build_nested_graph()

    result = nested.invoke({
        "data": "hello world",
        "processed": "",
        "validated": False,
        "messages": []
    })

    print(f"처리 결과: {result['processed']}")
    print(f"유효성 검사: {result['validated']}")


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 서브그래프 예제")
    print("=" * 60)

    run_subgraph_demo()
    run_nested_graph_demo()

    print("\n" + "=" * 60)
    print("핵심 개념:")
    print("1. 서브그래프 = 컴파일된 StateGraph")
    print("2. add_node(이름, 서브그래프)로 메인 그래프에 삽입")
    print("3. 공통 State 키로 메인↔서브 통신")
    print("4. 서브그래프 안에 서브그래프 중첩 가능")
    print("=" * 60)
