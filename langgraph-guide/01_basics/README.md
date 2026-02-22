# 01. LangGraph 기초

## LangGraph란?

LangGraph는 **상태 머신(State Machine)** 기반의 AI 에이전트 프레임워크입니다.
LLM을 활용한 복잡한 워크플로우를 **그래프** 형태로 표현하고 실행할 수 있게 해줍니다.

```
[노드 A] --엣지--> [노드 B] --엣지--> [노드 C]
```

## 핵심 개념

### 1. 그래프 (Graph)
- 전체 워크플로우를 나타내는 컨테이너
- **StateGraph**: 상태를 공유하는 노드들의 집합
- `START` → 노드들 → `END` 로 흐름이 구성됨

### 2. 상태 (State)
- 그래프 실행 중 노드 간에 공유되는 **데이터**
- Python의 `TypedDict`로 정의
- 각 노드는 상태를 받아서 수정된 상태를 반환

### 3. 노드 (Node)
- 실제 작업을 수행하는 **함수**
- 입력: 현재 State
- 출력: 업데이트할 State의 일부

### 4. 엣지 (Edge)
- 노드 간의 **연결**을 정의
- 일반 엣지: 항상 같은 방향으로 이동
- 조건부 엣지: 상태에 따라 다른 노드로 분기

## 파일 구성

| 파일 | 내용 |
|------|------|
| `01_hello_graph.py` | 첫 번째 그래프 만들기 - 기본 구조 이해 |
| `02_state_management.py` | State 설계와 Reducer 함수 이해 |
| `03_simple_chatbot.py` | LLM을 활용한 간단한 챗봇 구현 |

## 실행 방법

```bash
# 환경변수 설정 후 실행
python 01_hello_graph.py
python 02_state_management.py
python 03_simple_chatbot.py
```

## LangGraph 아키텍처 다이어그램

```
┌─────────────────────────────────────────┐
│              StateGraph                  │
│                                          │
│   START                                  │
│     │                                    │
│     ▼                                    │
│  [Node A] ──────────────▶ [Node B]       │
│     │                        │           │
│     │                        ▼           │
│     └──────────────────▶ [Node C]        │
│                              │           │
│                              ▼           │
│                             END          │
│                                          │
│  State = { key1: val1, key2: val2 }     │
└─────────────────────────────────────────┘
```
