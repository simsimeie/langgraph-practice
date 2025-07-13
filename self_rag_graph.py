#%%
from dotenv import load_dotenv

load_dotenv()
#%%
# SQLite3 버전 문제 해결: Chroma는 SQLite3 3.35.0 이상이 필요하지만 시스템에는 낮은 버전이 설치되어 있음
# pysqlite3-binary를 사용하여 더 높은 버전의 SQLite3을 제공
import sys
import pysqlite3
# 기존 sqlite3 모듈을 pysqlite3으로 대체
sys.modules['sqlite3'] = pysqlite3
print(f"Using SQLite version: {pysqlite3.sqlite_version}")
#%%
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ChromaDB 클라이언트를 생성합니다.
client = chromadb.PersistentClient(path='./brief')


embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')

# 컬랙션이 이미 존재할 때는 Chroma 클래스 생성자 사용하여 객체 생성
vector_store = Chroma(
    client=client,
    embedding_function=embedding_function,
    collection_name = 'brief'
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#%%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str
#%%
def retrieve(state: AgentState):
    query = state['query']
    docs = retriever.invoke(query)
    return {'context': docs}
#%%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4o")
#%%
# Create a LANGSMITH_API_KEY in Settings > API Keys
from langsmith import Client
from langchain_openai import ChatOpenAI

client = Client()
generate_prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
llm = ChatOpenAI(model="gpt-4o")
#%%
def generate(state: AgentState):
    context = state['context']
    query = state['query']
    rag_chain = generate_prompt | llm
    response = rag_chain.invoke({'context': context, 'question': query})
    return {'answer': response}
#%%
# Create a LANGSMITH_API_KEY in Settings > API Keys
from langsmith import Client
client = Client()
relevance_doc_prompt = client.pull_prompt("langchain-ai/rag-document-relevance", include_model=True)
#%%
from typing import Literal

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
    query = state['query']
    context = state['context']
    relevance_doc_chain = relevance_doc_prompt | llm
    response = relevance_doc_chain.invoke({'question': query, 'documents': context})
    if response['Score'] == 1:
        return 'relevant'

    return 'irrelevant'
#%%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['LLM -> 생성형AI', '돈 관련 단위 -> 달러', '엔스로픽 -> 엔스롭픽']

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고 우리의 사전을 참고해서 사용자의 질문을 변경해주세요
사전 : {dictionary}
질문 : {{query}}
                                              """)

def rewrite(state: AgentState):
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    response = rewrite_chain.invoke({'query': query})
    return {'query': response}
#%%
# Create a LANGSMITH_API_KEY in Settings > API Keys
from langsmith import Client
from langchain_core.prompts import PromptTemplate
client = Client()
hallucination_promt = client.pull_prompt("langchain-ai/rag-answer-hallucination", include_model=True)


def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']
    context = state['context']
    print(f'context == {context}')
    hallucination_chain = hallucination_promt | llm
    response = hallucination_chain.invoke({'student_answer': answer, 'documents': context})
    print(f'hallucination response : {response}')
    if response['Score'] == 1:
        return 'not hallucinated'

    return 'hallucinated'
#%%
query = '구글이 엔스로픽에 투자한 금액은 얼마인가요?'
context = retriever.invoke(query)

generate_state = {'query': query, 'context': context}
answer = generate(generate_state)

print(f'answer == {answer}')

hallucination_state = {'answer': answer, 'context': context}
check_result = check_hallucination(hallucination_state)
#%%
# Create a LANGSMITH_API_KEY in Settings > API Keys
from langsmith import Client
helpfulness_prompt = client.pull_prompt("langchain-ai/rag-answer-helpfulness", include_model=True)

def check_helpfulness_grader(state: AgentState) -> str:
    """
    사용자의 질문에 기반하여 생성된 답변의 유용성을 평가합니다.

    Args:
        state (AgentState): 사용자의 질문과 생성된 답변을 포함한 에이전트의 현재 state.

    Returns:
        str: 답변이 유용하다고 판단되면 'helpful', 그렇지 않으면 'unhelpful'을 반환합니다.
    """
    # state에서 질문과 답변을 추출합니다
    query = state['query']
    answer = state['answer']

    # 답변의 유용성을 평가하기 위한 체인을 생성합니다
    helpfulness_chain = helpfulness_prompt | llm

    # 질문과 답변으로 체인을 호출합니다
    response = helpfulness_chain.invoke({'question': query, 'student_answer': answer})

    # 점수가 1이면 'helpful'을 반환하고, 그렇지 않으면 'unhelpful'을 반환합니다
    if response['Score'] == 1:
        return 'helpful'

    return 'unhelpful'


# 노드는 state를 반환해야하기 때문에 아래와 같이 state만 반환하도록 만들었다.
def check_helpfulness(state: AgentState) -> AgentState:
    """
    유용성을 확인하는 자리 표시자 함수입니다.
    graph에서 conditional_edge를 연속으로 사용하지 않고 node를 추가해
    가독성을 높이기 위해 사용합니다

    Args:
        state (AgentState): 에이전트의 현재 state.

    Returns:
        AgentState: 변경되지 않은 state를 반환합니다.
    """
    # 이 함수는 현재 아무 작업도 수행하지 않으며 state를 그대로 반환합니다
    return state

#%%
query = '구글이 엔스로픽에 투자한 금액은 얼마인가요?'
context = retriever.invoke(query)

generate_state = {'query': query, 'context': context}
answer = generate(generate_state)

print(f'answer == {answer}')

helpfulness_state = {'answer': answer, 'query': query}
check_helpfulness(helpfulness_state)
#%%
from langgraph.graph import StateGraph

graph_builder = StateGraph(AgentState)
#%%
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)
#%%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)
graph_builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'not hallucinated': 'check_helpfulness',
        'hallucinated': 'generate'
    }
)

graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)
graph_builder.add_edge('rewrite', 'retrieve')
#%%
graph = graph_builder.compile()
