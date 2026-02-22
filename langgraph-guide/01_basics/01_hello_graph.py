"""
LangGraph 첫 번째 예제: Hello Graph
=====================================

목표: LangGraph의 가장 기본적인 구조를 이해한다.

핵심 개념:
- StateGraph: 상태를 공유하는 그래프 컨테이너
- TypedDict: State(상태) 정의에 사용되는 Python 타입
- Node: 실제 작업을 수행하는 함수
- Edge: 노드 간 연결
- START / END: 그래프의 진입점과 종료점
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================
# STEP 1: State(상태) 정의
# ============================================================
# State는 그래프 실행 중 모든 노드가 공유하는 데이터 컨테이너입니다.
# TypedDict를 사용해서 타입 안전성을 보장합니다.
#
# 비유: State는 회사의 공유 문서 같은 것입니다.
#       여러 부서(노드)가 이 문서를 읽고 수정합니다.

class SimpleState(TypedDict):
    # 메시지 문자열을 담는 필드
    message: str
    # 처리 횟수를 추적하는 필드
    count: int


# ============================================================
# STEP 2: Node(노드) 정의
# ============================================================
# 노드는 State를 받아서 State의 일부를 업데이트해 반환하는 순수 함수입니다.
#
# 중요: 노드는 State 전체를 반환할 필요가 없습니다.
#       변경하고 싶은 키만 딕셔너리로 반환하면 됩니다.
#       나머지 키는 LangGraph가 자동으로 유지합니다.

def greet_node(state: SimpleState) -> dict:
    """
    인사말을 추가하는 노드.

    Args:
        state: 현재 그래프 상태
    Returns:
        업데이트할 상태 필드 (딕셔너리)
    """
    print(f"[greet_node] 실행 중... 현재 메시지: {state['message']}")

    # state에서 기존 메시지를 읽고, 새 메시지로 업데이트
    updated_message = f"안녕하세요! {state['message']}"

    # 변경하고 싶은 필드만 반환 (count는 건드리지 않음)
    return {"message": updated_message}


def process_node(state: SimpleState) -> dict:
    """
    메시지를 처리하고 카운트를 증가시키는 노드.
    """
    print(f"[process_node] 실행 중... 현재 카운트: {state['count']}")

    # 메시지에 처리 완료 표시 추가
    processed_message = f"{state['message']} (처리 완료)"

    # 카운트 1 증가
    new_count = state['count'] + 1

    return {
        "message": processed_message,
        "count": new_count
    }


def finalize_node(state: SimpleState) -> dict:
    """
    최종 결과를 정리하는 노드.
    """
    print(f"[finalize_node] 실행 중... 최종 정리 중")

    final_message = f"[최종] {state['message']} | 총 {state['count']}번 처리됨"

    return {"message": final_message}


# ============================================================
# STEP 3: Graph 구성
# ============================================================
# StateGraph를 생성하고 노드와 엣지를 추가합니다.

def build_graph():
    """
    그래프를 구성하고 컴파일된 그래프를 반환합니다.

    그래프 구조:
    START → greet → process → finalize → END
    """
    # StateGraph 생성 - State 타입을 인자로 전달
    # 이렇게 하면 LangGraph가 State의 구조를 알 수 있습니다.
    graph_builder = StateGraph(SimpleState)

    # ── 노드 추가 ──────────────────────────────────────────
    # add_node(이름, 함수)
    # 이름은 그래프 내에서 노드를 식별하는 고유한 문자열입니다.
    graph_builder.add_node("greet", greet_node)
    graph_builder.add_node("process", process_node)
    graph_builder.add_node("finalize", finalize_node)

    # ── 엣지 추가 ──────────────────────────────────────────
    # add_edge(from_node, to_node)
    # 노드 간의 실행 순서를 정의합니다.

    # START → greet: 그래프 시작점에서 greet 노드로
    graph_builder.add_edge(START, "greet")

    # greet → process: greet 완료 후 process 노드로
    graph_builder.add_edge("greet", "process")

    # process → finalize: process 완료 후 finalize 노드로
    graph_builder.add_edge("process", "finalize")

    # finalize → END: finalize 완료 후 그래프 종료
    graph_builder.add_edge("finalize", END)

    # ── 그래프 컴파일 ──────────────────────────────────────
    # compile()을 호출해야 실행 가능한 그래프가 됩니다.
    # 이 시점에서 그래프의 유효성 검사가 이루어집니다.
    graph = graph_builder.compile()

    return graph


# ============================================================
# STEP 4: Graph 실행
# ============================================================

def main():
    # 그래프 빌드
    graph = build_graph()

    # 그래프 구조 시각화 (Mermaid 다이어그램)
    # 실제 환경에서는 draw_mermaid_png()로 이미지 출력 가능
    print("=== 그래프 구조 (Mermaid) ===")
    print(graph.get_graph().draw_mermaid())
    print()

    # ── 초기 State 설정 ────────────────────────────────────
    # invoke()에 초기 상태를 딕셔너리로 전달합니다.
    initial_state = {
        "message": "LangGraph 학습 중",
        "count": 0
    }

    print("=== 그래프 실행 ===")
    print(f"초기 상태: {initial_state}")
    print()

    # ── 그래프 실행 ────────────────────────────────────────
    # invoke()는 그래프를 동기적으로 실행하고 최종 State를 반환합니다.
    # START → greet → process → finalize → END 순서로 실행됩니다.
    final_state = graph.invoke(initial_state)

    print()
    print("=== 최종 상태 ===")
    print(f"메시지: {final_state['message']}")
    print(f"카운트: {final_state['count']}")

    # ── 다른 입력으로 다시 실행 ────────────────────────────
    print("\n=== 두 번째 실행 ===")
    result2 = graph.invoke({
        "message": "두 번째 테스트",
        "count": 5
    })
    print(f"최종 메시지: {result2['message']}")


# ============================================================
# 추가 학습: 그래프 검사 메서드들
# ============================================================
def explore_graph():
    """그래프의 다양한 정보를 확인하는 예제"""
    graph = build_graph()

    # 노드 목록 확인
    print("노드 목록:", list(graph.get_graph().nodes.keys()))

    # 엣지 목록 확인
    print("엣지 목록:", graph.get_graph().edges)

    # 입력 스키마 확인 (어떤 키가 필요한지)
    print("입력 스키마:", graph.input_schema.schema())

    # 출력 스키마 확인
    print("출력 스키마:", graph.output_schema.schema())


if __name__ == "__main__":
    main()
    print("\n" + "="*50)
    print("그래프 메타데이터:")
    explore_graph()
