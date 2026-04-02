import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 페이지 설정
st.set_page_config(page_title="한세대학교 규정 챗봇", page_icon="🏫")
st.title("🏫 한세대학교 규정 안내 챗봇")
st.markdown("규정집에 대해 궁금한 점을 물어보세요.")

# API 키 및 DB 설정
os.environ["OPENAI_API_KEY"] = "sk-proj-7Slo88jqTeWEI-l0_EJJHzyjkDaICSYzj01HsJvdPafeeG4k3zHd_WfKUHiofquEBzjO-F5wS4T3BlbkFJHKi_-JYaKChGbhd6q5HcYiL5iO5-8PvifpoJhlx561Ureoh7kajDUQ4hQpRJY1f74FPXOoGmEAsk-proj-YcydfEcnon-pPOlANUM4W5B-u-TvEa6M9qHAIEfGePyWgi8prRwBs3o8Jhdm8rwnf0guRtLA0IT3BlbkFJyM8XRzlLbE46RtQJwuFIr_k4F0hG5o0VYOlZkSyFMZ6xh1QlsNkECNXpqLONcVRKjSbAJl-QAA"
persist_db = "./db_hansei"

# 데이터베이스 불러오기 (이미 생성된 DB 활용)
@st.cache_resource
def load_db():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=persist_db, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = load_db()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 챗봇 로직
template = """당신은 한세대학교 규정 전문가입니다. 문맥을 바탕으로 답변하고 페이지를 명시하세요.
{context}
질문: {question}
답변:"""
prompt = ChatPromptTemplate.from_template(template)
chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# 채팅 UI 구현
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("규정집 분석 중..."):
            response = chain.invoke(prompt_input)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
