# 02. 엣지(Edge)와 라우팅(Routing)

## 개요

LangGraph에서 **엣지(Edge)**는 노드 간의 실행 흐름을 정의합니다.
단순히 순서대로 실행하는 것 외에도, **조건에 따라 다른 경로**로 분기할 수 있습니다.

## 엣지의 종류

### 1. 일반 엣지 (Normal Edge)
항상 같은 방향으로 이동합니다.

```python
graph.add_edge("node_a", "node_b")  # a → b 항상 실행
```

### 2. 조건부 엣지 (Conditional Edge)
State를 분석해서 어느 노드로 갈지 결정합니다.

```python
def route_function(state) -> str:
    if state["score"] > 90:
        return "excellent_node"
    else:
        return "normal_node"

graph.add_conditional_edges(
    "evaluator_node",  # 이 노드가 끝난 후
    route_function,    # 이 함수로 다음 노드 결정
)
```

### 3. 멀티 소스 엣지
여러 노드에서 같은 노드로 연결합니다.

```python
# 여러 노드 → 하나의 노드
for source in ["node_a", "node_b", "node_c"]:
    graph.add_edge(source, "join_node")
```

## 라우팅 패턴

```
┌─────────────────────────────────────────────────┐
│            조건부 라우팅 예시                      │
│                                                   │
│         [분류기 노드]                              │
│              │                                    │
│    ┌─────────┼─────────┐                         │
│    │         │         │                          │
│    ▼         ▼         ▼                          │
│ [기술질문] [일반질문] [불분명]                      │
│    │         │         │                          │
│    └─────────┼─────────┘                         │
│              │                                    │
│           [응답 노드]                              │
└─────────────────────────────────────────────────┘
```

## 파일 구성

| 파일 | 내용 |
|------|------|
| `01_conditional_edges.py` | 조건부 엣지 기본 사용법 |
| `02_routing_patterns.py` | 실전 라우팅 패턴 모음 |
