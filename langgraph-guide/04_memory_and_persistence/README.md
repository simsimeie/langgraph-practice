# 04. 메모리(Memory)와 지속성(Persistence)

## 개요

LangGraph는 **체크포인터(Checkpointer)**를 통해 그래프 실행 상태를 저장합니다.
이를 활용하면:
- 대화 기록이 자동으로 유지됩니다 (멀티턴 대화)
- 그래프 실행을 일시 중단하고 나중에 재개할 수 있습니다
- 에러 발생 시 마지막 저장 지점으로 복구할 수 있습니다

## 핵심 개념

### Checkpointer
그래프의 **State 스냅샷**을 저장하는 컴포넌트입니다.

```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)
```

### Thread ID
각 독립적인 대화 세션을 구분하는 **식별자**입니다.

```python
config = {"configurable": {"thread_id": "user_123"}}
graph.invoke({"messages": [...]}, config=config)
```

같은 thread_id를 사용하면 이전 대화 기록이 이어집니다.
다른 thread_id를 사용하면 새로운 대화가 시작됩니다.

### 체크포인터 종류

| 체크포인터 | 저장 위치 | 특징 |
|-----------|---------|------|
| `MemorySaver` | 인메모리 | 빠름, 프로세스 종료 시 소멸 |
| `SqliteSaver` | SQLite 파일 | 영구 저장, 단일 서버용 |
| `PostgresSaver` | PostgreSQL | 영구 저장, 분산 환경용 |

## Thread 격리

```
Thread "user_A"          Thread "user_B"
     │                        │
[메시지1: 안녕]          [메시지1: Hello]
     │                        │
[메시지2: 날씨?]         [메시지2: Weather?]
     │                        │
     ↓                        ↓
 (완전히 독립적인 대화 기록)
```

## 파일 구성

| 파일 | 내용 |
|------|------|
| `01_memory_saver.py` | MemorySaver를 사용한 멀티턴 대화 |
| `02_sqlite_checkpointer.py` | SQLite로 영구 저장 및 재개 |
