# 06. 스트리밍 (Streaming)

## 개요

LangGraph는 그래프 실행 결과를 **실시간으로 스트리밍**할 수 있습니다.
사용자는 LLM 응답이 완성되기를 기다리지 않고, 생성되는 즉시 볼 수 있습니다.

## stream_mode 종류

| stream_mode | 설명 | 반환 형식 |
|-------------|------|-----------|
| `"values"` | 각 노드 실행 후 전체 State 반환 | `state_dict` |
| `"updates"` | 각 노드의 변경사항만 반환 | `{node_name: changes}` |
| `"messages"` | 메시지를 토큰 단위로 스트리밍 | `(message_chunk, metadata)` |
| `"debug"` | 모든 내부 이벤트 반환 (디버깅용) | 상세 이벤트 객체 |

## stream_mode 선택 가이드

```
토큰별 실시간 출력이 필요한가?
  → stream_mode="messages"

노드별 진행 상황을 추적하고 싶은가?
  → stream_mode="updates"

각 단계의 전체 상태가 필요한가?
  → stream_mode="values"

내부 동작을 디버깅하고 싶은가?
  → stream_mode="debug"
```

## astream() - 비동기 스트리밍

FastAPI, asyncio 환경에서는 `astream()`을 사용합니다:

```python
async for event in graph.astream(input, config=config):
    print(event)
```

## 파일 구성

| 파일 | 내용 |
|------|------|
| `01_streaming_modes.py` | 모든 stream_mode 사용법 예제 |
