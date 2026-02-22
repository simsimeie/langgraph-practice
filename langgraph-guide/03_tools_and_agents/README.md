# 03. 도구(Tools)와 에이전트(Agents)

## 개요

LangGraph에서 **도구(Tool)**는 LLM이 호출할 수 있는 외부 함수입니다.
검색, 계산, API 호출, 데이터베이스 조회 등을 LLM이 스스로 결정하여 실행합니다.

## 핵심 개념

### ToolNode
LangGraph에서 제공하는 도구 실행 전용 노드입니다.

```python
from langgraph.prebuilt import ToolNode

tools = [search_tool, calculator_tool]
tool_node = ToolNode(tools)
```

### 도구 바인딩 (Tool Binding)
LLM에게 어떤 도구를 사용할 수 있는지 알려줍니다.

```python
llm_with_tools = llm.bind_tools(tools)
```

### ReAct 패턴
**Re**asoning + **Act**ing = LLM이 추론하고 도구를 호출하는 반복 패턴

```
생각 → 도구 호출 → 결과 확인 → 다시 생각 → ... → 최종 답변
```

## 에이전트 동작 흐름

```
START
  │
  ▼
[agent 노드]  ← LLM이 "도구를 써야겠다" 판단
  │
  ├─ 도구 필요 ──▶ [tool 노드] ──▶ [agent 노드] (결과 보고 다시 판단)
  │
  └─ 완료 ────▶ END
```

## 파일 구성

| 파일 | 내용 |
|------|------|
| `01_tool_node.py` | @tool 데코레이터로 도구 만들기, ToolNode 사용법 |
| `02_react_agent.py` | ReAct 패턴 에이전트 구현 |
