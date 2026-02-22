"""
LangGraph Map-Reduce 패턴
===========================

목표: 큰 작업을 작은 단위로 나눠서 병렬 처리하고 결과를 합치는 패턴을 이해한다.

Map-Reduce란?
  Map: 큰 데이터를 작은 청크로 나눠서 독립적으로 처리
  Reduce: 처리된 결과들을 하나로 합침

언제 사용하나:
  - 긴 문서를 청크로 나눠서 각각 요약 후 최종 요약
  - 여러 데이터 소스를 각각 처리 후 통합
  - 대량의 항목을 병렬 처리 후 집계

LangGraph에서의 구현:
  Send API + operator.add Reducer의 조합으로 구현합니다.
"""

from typing import Annotated, List, TypedDict
import operator
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 예제 1: 문서 Map-Reduce 요약
# ============================================================
# 긴 문서를 청크로 나눠서 각각 요약하고, 최종 요약을 생성합니다.

class ChunkState(TypedDict):
    """각 청크를 처리하는 서브 State"""
    chunk_id: int
    chunk_text: str
    chunk_summary: str


class DocumentSummaryState(TypedDict):
    """전체 문서 처리 메인 State"""
    full_document: str
    # Map 단계: 각 청크의 요약이 여기에 모임
    chunk_summaries: Annotated[List[str], operator.add]
    # Reduce 단계: 최종 통합 요약
    final_summary: str
    # 처리 메타데이터
    total_chunks: int


def split_document_node(state: DocumentSummaryState) -> List[Send]:
    """
    MAP 단계 1: 문서를 청크로 분할하고 병렬 처리 시작.

    각 청크에 대해 Send 이벤트를 생성해서 summarize_chunk 노드를 병렬 실행합니다.
    """
    document = state["full_document"]

    # 문서를 문장 단위로 분할 (실제로는 토큰 수 기반으로 분할)
    sentences = document.split(". ")
    # 2문장씩 하나의 청크로 묶기
    chunks = []
    for i in range(0, len(sentences), 2):
        chunk = ". ".join(sentences[i:i+2])
        if chunk:
            chunks.append(chunk)

    print(f"[split_document] 문서를 {len(chunks)}개 청크로 분할")

    # 각 청크에 대해 Send 이벤트 생성 (병렬 실행)
    return [
        Send("summarize_chunk", {
            "chunk_id": i,
            "chunk_text": chunk,
            "chunk_summary": ""
        })
        for i, chunk in enumerate(chunks)
    ]


def summarize_chunk_node(state: ChunkState) -> dict:
    """
    MAP 단계 2: 개별 청크를 요약 (병렬로 여러 번 실행됨).

    각 Send에 의해 독립적으로 실행됩니다.
    """
    chunk_id = state["chunk_id"]
    chunk_text = state["chunk_text"]

    # 실제로는 LLM으로 요약
    # summary_response = llm.invoke(f"다음 텍스트를 한 문장으로 요약하세요:\n{chunk_text}")
    # summary = summary_response.content

    # 시뮬레이션: 청크의 첫 50자
    summary = f"[청크 {chunk_id} 요약] {chunk_text[:60]}..."
    print(f"  [summarize_chunk] 청크 {chunk_id} 요약 완료")

    # chunk_summaries에 추가됨 (operator.add Reducer)
    return {"chunk_summaries": [summary]}


def combine_summaries_node(state: DocumentSummaryState) -> dict:
    """
    REDUCE 단계: 모든 청크 요약을 합쳐서 최종 요약 생성.

    모든 summarize_chunk가 완료된 후 한 번 실행됩니다.
    """
    summaries = state["chunk_summaries"]
    print(f"[combine_summaries] {len(summaries)}개 요약 통합 중")

    # 모든 요약을 합쳐서 LLM으로 최종 요약 생성
    combined_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))

    # 실제로는:
    # response = llm.invoke(f"다음 요약들을 하나의 단락으로 통합하세요:\n{combined_text}")
    # final_summary = response.content

    final_summary = f"최종 통합 요약 ({len(summaries)}개 섹션):\n" + combined_text

    return {
        "final_summary": final_summary,
        "total_chunks": len(summaries)
    }


def build_map_reduce_graph():
    """
    Map-Reduce 문서 요약 그래프.

    구조:
    START
      │
      ▼
    split_document (문서 분할 + Send 이벤트 생성)
      │
      ├──▶ summarize_chunk (청크 0)
      ├──▶ summarize_chunk (청크 1)  ← 병렬 실행
      └──▶ summarize_chunk (청크 N)
              │
              ▼ (모두 완료 후)
         combine_summaries (최종 요약)
              │
              ▼
             END
    """
    builder = StateGraph(DocumentSummaryState)

    builder.add_node("split_document", split_document_node)
    builder.add_node("summarize_chunk", summarize_chunk_node)
    builder.add_node("combine_summaries", combine_summaries_node)

    builder.add_edge(START, "split_document")
    # summarize_chunk는 Send에 의해 동적으로 실행됨
    builder.add_edge("summarize_chunk", "combine_summaries")
    builder.add_edge("combine_summaries", END)

    return builder.compile()


# ============================================================
# 예제 2: 다중 질문 병렬 처리 Map-Reduce
# ============================================================
# 여러 질문을 동시에 처리하고 결과를 통합합니다.

class QuestionState(TypedDict):
    """개별 질문 처리 State"""
    question_id: int
    question: str
    answer: str


class BatchQAState(TypedDict):
    """배치 질문 처리 메인 State"""
    questions: List[str]
    answers: Annotated[List[dict], operator.add]  # [{id, question, answer}, ...]
    final_report: str


def distribute_questions_node(state: BatchQAState) -> List[Send]:
    """각 질문을 병렬로 처리하는 Send 이벤트 생성"""
    print(f"[distribute_questions] {len(state['questions'])}개 질문 병렬 처리 시작")
    return [
        Send("answer_question", {
            "question_id": i,
            "question": q,
            "answer": ""
        })
        for i, q in enumerate(state["questions"])
    ]


def answer_question_node(state: QuestionState) -> dict:
    """개별 질문에 답변하는 노드 (병렬 실행)"""
    q_id = state["question_id"]
    question = state["question"]

    # 시뮬레이션 (실제로는 LLM 사용)
    answer = f"Q{q_id}: '{question[:30]}'에 대한 답변입니다."
    print(f"  [answer_question] Q{q_id} 답변 완료")

    return {"answers": [{"id": q_id, "question": question, "answer": answer}]}


def compile_report_node(state: BatchQAState) -> dict:
    """모든 답변을 정렬하고 보고서 생성"""
    # ID 순서로 정렬
    sorted_answers = sorted(state["answers"], key=lambda x: x["id"])

    report_lines = ["=== 질문-답변 보고서 ==="]
    for qa in sorted_answers:
        report_lines.append(f"\nQ{qa['id']}: {qa['question']}")
        report_lines.append(f"A{qa['id']}: {qa['answer']}")

    report = "\n".join(report_lines)
    print(f"[compile_report] {len(sorted_answers)}개 답변 통합 완료")
    return {"final_report": report}


def build_batch_qa_graph():
    """배치 질문 처리 Map-Reduce 그래프"""
    builder = StateGraph(BatchQAState)

    builder.add_node("distribute_questions", distribute_questions_node)
    builder.add_node("answer_question", answer_question_node)
    builder.add_node("compile_report", compile_report_node)

    builder.add_edge(START, "distribute_questions")
    builder.add_edge("answer_question", "compile_report")
    builder.add_edge("compile_report", END)

    return builder.compile()


# ============================================================
# 예제 3: 계층적 Map-Reduce
# ============================================================
# Map-Reduce를 여러 수준으로 적용합니다.
# 예: 책 → 챕터 → 단락 순서로 요약

class SectionState(TypedDict):
    section_id: int
    section_text: str
    section_summary: str


class ChapterState(TypedDict):
    chapter_id: int
    chapter_text: str
    section_summaries: Annotated[List[str], operator.add]
    chapter_summary: str


class BookState(TypedDict):
    chapters: List[str]
    chapter_summaries: Annotated[List[str], operator.add]
    book_summary: str


def process_chapters_node(state: BookState) -> List[Send]:
    """챕터별 병렬 처리"""
    print(f"[process_chapters] {len(state['chapters'])}개 챕터 처리")
    return [
        Send("summarize_chapter_node", {
            "chapter_id": i,
            "chapter_text": ch,
            "section_summaries": [],
            "chapter_summary": ""
        })
        for i, ch in enumerate(state["chapters"])
    ]


def summarize_chapter_node(state: ChapterState) -> dict:
    """챕터 요약"""
    summary = f"챕터 {state['chapter_id']} 요약: {state['chapter_text'][:40]}..."
    print(f"  [summarize_chapter] 챕터 {state['chapter_id']} 완료")
    return {"chapter_summaries": [summary]}


def write_book_summary_node(state: BookState) -> dict:
    """전체 책 요약"""
    book_summary = "전체 책 요약:\n" + "\n".join(state["chapter_summaries"])
    return {"book_summary": book_summary}


def build_hierarchical_map_reduce():
    """계층적 Map-Reduce 그래프"""
    builder = StateGraph(BookState)
    builder.add_node("process_chapters", process_chapters_node)
    builder.add_node("summarize_chapter_node", summarize_chapter_node)
    builder.add_node("write_book_summary", write_book_summary_node)

    builder.add_edge(START, "process_chapters")
    builder.add_edge("summarize_chapter_node", "write_book_summary")
    builder.add_edge("write_book_summary", END)

    return builder.compile()


# ============================================================
# 실행
# ============================================================

SAMPLE_DOCUMENT = """LangGraph는 LLM 기반 에이전트를 위한 프레임워크입니다.
상태 머신 개념을 사용합니다.
LangChain 위에서 동작하며 복잡한 워크플로우를 쉽게 구현할 수 있습니다.
Python으로 작성되었으며 타입 안전성을 보장합니다.
노드, 엣지, 상태의 세 가지 핵심 개념으로 구성됩니다."""

SAMPLE_QUESTIONS = [
    "Python의 주요 특징은 무엇인가요?",
    "머신러닝과 딥러닝의 차이점은?",
    "LangGraph는 언제 사용하나요?",
    "API와 SDK의 차이점은?",
]

SAMPLE_CHAPTERS = [
    "제1장: LangGraph 소개와 기본 개념",
    "제2장: 노드와 엣지 설계 방법",
    "제3장: 상태 관리와 체크포인터",
    "제4장: 고급 기능과 실전 패턴",
]


if __name__ == "__main__":
    print("=" * 60)
    print("Map-Reduce 패턴 예제")
    print("=" * 60)

    # 예제 1: 문서 요약
    print("\n[예제 1: 문서 Map-Reduce 요약]")
    doc_graph = build_map_reduce_graph()
    result = doc_graph.invoke({
        "full_document": SAMPLE_DOCUMENT,
        "chunk_summaries": [],
        "final_summary": "",
        "total_chunks": 0
    })
    print(f"\n처리된 청크 수: {result['total_chunks']}")
    print(f"최종 요약:\n{result['final_summary'][:300]}")

    # 예제 2: 배치 질문 처리
    print("\n" + "-" * 60)
    print("[예제 2: 배치 질문 병렬 처리]")
    qa_graph = build_batch_qa_graph()
    result = qa_graph.invoke({
        "questions": SAMPLE_QUESTIONS,
        "answers": [],
        "final_report": ""
    })
    print(f"\n{result['final_report']}")

    # 예제 3: 계층적 Map-Reduce
    print("\n" + "-" * 60)
    print("[예제 3: 계층적 Map-Reduce (책 요약)]")
    book_graph = build_hierarchical_map_reduce()
    result = book_graph.invoke({
        "chapters": SAMPLE_CHAPTERS,
        "chapter_summaries": [],
        "book_summary": ""
    })
    print(f"\n{result['book_summary']}")
