# 05. Human-in-the-Loop (휴먼 인 더 루프)

## 개요

**Human-in-the-Loop**는 그래프 실행 중에 사람이 개입하여 검토, 승인, 수정할 수 있는 패턴입니다.

자율 에이전트가 중요한 작업(파일 삭제, 결제, 이메일 발송 등)을 수행하기 전에 사람의 확인을 받을 수 있습니다.

## interrupt() 동작 원리

```
그래프 실행
    │
    ▼
[노드 A 실행]
    │
    ▼
[interrupt() 호출] ← 여기서 실행이 일시 정지됨
    │
    │ ← 사람이 검토하고 Command로 재개/수정
    ▼
[노드 B 실행 (재개)]
    │
    ▼
   END
```

## 핵심 API

### interrupt()
노드 내에서 실행을 일시 정지하고 사람에게 제어를 넘깁니다.

```python
from langgraph.types import interrupt

def my_node(state):
    user_feedback = interrupt({
        "message": "이 작업을 승인하시겠습니까?",
        "data": state["action_data"]
    })
    # 사람이 Command로 재개하면 여기서 user_feedback 값을 받음
    if user_feedback == "approved":
        return {"status": "approved"}
```

### Command
실행을 재개하거나 State를 수정하는 명령입니다.

```python
from langgraph.types import Command

# 단순 재개
graph.invoke(Command(resume="approved"), config=config)

# State 수정 후 재개
graph.invoke(Command(resume="approved", update={"key": "value"}), config=config)
```

## 주의사항
- interrupt()는 반드시 **체크포인터**가 있어야 작동합니다
- 재개 시에는 동일한 `thread_id`와 `config`를 사용해야 합니다

## 파일 구성

| 파일 | 내용 |
|------|------|
| `01_interrupt_basic.py` | interrupt() 기본 사용법 |
| `02_approval_workflow.py` | 승인 워크플로우 실전 예제 |
