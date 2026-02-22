# LangGraph 1.x 완전 학습 가이드

> LangGraph의 기초부터 고급 기능까지, 체계적으로 배우는 실습 코드 모음

## 시작하기 전에

### 환경 설정

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일을 열고 OPENAI_API_KEY를 입력하세요
```

### 디렉토리 구조

```
langgraph-guide/
├── README.md                       ← 지금 이 파일
├── requirements.txt
├── .env.example
│
├── 01_basics/                      ← 🟢 기초 (여기서 시작하세요)
│   ├── README.md
│   ├── 01_hello_graph.py           # 첫 번째 그래프
│   ├── 02_state_management.py      # State와 Reducer
│   └── 03_simple_chatbot.py        # LLM 챗봇
│
├── 02_edges_and_routing/           ← 🟢 엣지와 라우팅
│   ├── README.md
│   ├── 01_conditional_edges.py     # 조건부 엣지
│   └── 02_routing_patterns.py      # 실전 라우팅 패턴
│
├── 03_tools_and_agents/            ← 🟡 도구와 에이전트
│   ├── README.md
│   ├── 01_tool_node.py             # @tool과 ToolNode
│   └── 02_react_agent.py           # ReAct 에이전트
│
├── 04_memory_and_persistence/      ← 🟡 메모리와 지속성
│   ├── README.md
│   ├── 01_memory_saver.py          # MemorySaver
│   └── 02_sqlite_checkpointer.py   # SQLite 영구 저장
│
├── 05_human_in_the_loop/           ← 🟡 휴먼 인 더 루프
│   ├── README.md
│   ├── 01_interrupt_basic.py       # interrupt() 기본
│   └── 02_approval_workflow.py     # 승인 워크플로우
│
├── 06_streaming/                   ← 🟡 스트리밍
│   ├── README.md
│   └── 01_streaming_modes.py       # 모든 스트리밍 모드
│
├── 07_advanced/                    ← 🔴 고급 기능
│   ├── README.md
│   ├── 01_subgraphs.py             # 서브그래프
│   ├── 02_parallel_nodes.py        # 병렬 실행 / Send API
│   └── 03_map_reduce.py            # Map-Reduce 패턴
│
└── 08_multi_agent/                 ← 🔴 멀티 에이전트
    ├── README.md
    ├── 01_supervisor_pattern.py    # 수퍼바이저 패턴
    └── 02_handoff_pattern.py       # 핸드오프 패턴
```

---

## 학습 로드맵

### Phase 1: 기초 이해 (🟢)

| 순서 | 파일 | 핵심 개념 |
|------|------|-----------|
| 1 | `01_basics/01_hello_graph.py` | StateGraph, Node, Edge |
| 2 | `01_basics/02_state_management.py` | Reducer, Annotated |
| 3 | `01_basics/03_simple_chatbot.py` | LLM 연동, add_messages |
| 4 | `02_edges_and_routing/01_conditional_edges.py` | add_conditional_edges |
| 5 | `02_edges_and_routing/02_routing_patterns.py` | 폴백, 게이트키퍼 |

### Phase 2: 핵심 기능 (🟡)

| 순서 | 파일 | 핵심 개념 |
|------|------|-----------|
| 6 | `03_tools_and_agents/01_tool_node.py` | @tool, ToolNode |
| 7 | `03_tools_and_agents/02_react_agent.py` | ReAct 패턴 |
| 8 | `04_memory_and_persistence/01_memory_saver.py` | MemorySaver, thread_id |
| 9 | `04_memory_and_persistence/02_sqlite_checkpointer.py` | 영구 저장 |
| 10 | `05_human_in_the_loop/01_interrupt_basic.py` | interrupt(), Command |
| 11 | `05_human_in_the_loop/02_approval_workflow.py` | 승인 워크플로우 |
| 12 | `06_streaming/01_streaming_modes.py` | stream_mode |

### Phase 3: 고급 기능 (🔴)

| 순서 | 파일 | 핵심 개념 |
|------|------|-----------|
| 13 | `07_advanced/01_subgraphs.py` | 서브그래프, 모듈화 |
| 14 | `07_advanced/02_parallel_nodes.py` | Send API, Fan-out |
| 15 | `07_advanced/03_map_reduce.py` | Map-Reduce 패턴 |
| 16 | `08_multi_agent/01_supervisor_pattern.py` | Supervisor |
| 17 | `08_multi_agent/02_handoff_pattern.py` | Handoff, Command goto |

---

## 핵심 개념 요약

### 1. StateGraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class MyState(TypedDict):
    value: str

graph = StateGraph(MyState)
graph.add_node("my_node", my_function)
graph.add_edge(START, "my_node")
graph.add_edge("my_node", END)
compiled = graph.compile()
result = compiled.invoke({"value": "hello"})
```

### 2. Reducer

```python
from typing import Annotated
import operator
from langgraph.graph.message import add_messages

class State(TypedDict):
    # 덮어쓰기 (기본)
    name: str

    # 리스트 이어붙이기
    items: Annotated[list, operator.add]

    # 메시지 스마트 병합
    messages: Annotated[list, add_messages]
```

### 3. 조건부 엣지

```python
def router(state) -> str:
    if state["score"] > 90:
        return "excellent"
    return "normal"

graph.add_conditional_edges("evaluator", router)
```

### 4. 체크포인터 (메모리)

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())

# thread_id로 대화 구분
config = {"configurable": {"thread_id": "user_123"}}
graph.invoke(state, config=config)
```

### 5. Human-in-the-Loop

```python
from langgraph.types import interrupt, Command

def review_node(state):
    decision = interrupt({"message": "승인하시겠습니까?"})
    return {"approved": decision == "approve"}

# 재개
graph.invoke(Command(resume="approve"), config=config)
```

### 6. Send API (병렬 실행)

```python
from langgraph.types import Send

def distribute(state) -> list[Send]:
    return [
        Send("process_item", {"item": item})
        for item in state["items"]
    ]
```

---

## 자주 쓰는 임포트 치트시트

```python
# 핵심 그래프
from langgraph.graph import StateGraph, START, END

# 체크포인터
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver  # pip install langgraph-checkpoint-sqlite

# 메시지 관리
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# 도구
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

# 고급 기능
from langgraph.types import interrupt, Command, Send

# 내장 에이전트
from langgraph.prebuilt import create_react_agent

# 타입
from typing import Annotated, List, Literal, TypedDict
import operator
```

---

## 일반적인 그래프 패턴

### 단순 챗봇
```
START → chatbot → END
```

### ReAct 에이전트
```
START → agent → (도구 필요?) → tools → agent → ... → END
```

### 승인 워크플로우
```
START → prepare → [interrupt] → execute → END
```

### 수퍼바이저 멀티 에이전트
```
START → supervisor → agent_A → supervisor → agent_B → ... → END
```

### Map-Reduce
```
START → split → [process × N (병렬)] → combine → END
```

---

## 학습 팁

1. **순서대로 공부하세요**: Phase 1 → 2 → 3 순서로 진행하면 이해가 쉽습니다.
2. **코드를 직접 실행하세요**: 주석을 읽고 실행해보면서 확인하세요.
3. **README.md를 먼저 읽으세요**: 각 챕터의 README에 개념 설명이 있습니다.
4. **LangSmith를 활용하세요**: `.env`에 LangSmith 설정을 추가하면 그래프 실행을 시각적으로 추적할 수 있습니다.
5. **공식 문서를 참고하세요**: https://langchain-ai.github.io/langgraph/

---

## 버전 정보

- LangGraph: 1.x
- LangChain: 0.3.x
- Python: 3.9+
