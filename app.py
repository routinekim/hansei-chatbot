import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="한세대학교 규정 챗봇", page_icon="🏫")
st.title("🏫 한세대학교 규정 안내 챗봇")

# 1. API 키 설정 (Secrets 활용)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Secrets에 OPENAI_API_KEY를 등록해주세요.")
    st.stop()

# 2. 데이터 로드 및 초기화 (PDF 직접 읽기)
@st.cache_resource
def prepare_database():
    pdf_path = "한세대규정_전체.pdf"  # 깃허브에 올린 PDF 파일명과 정확히 일치해야 함
    
    if not os.path.exists(pdf_path):
        st.error(f"❌ '{pdf_path}' 파일을 찾을 수 없습니다. 깃허브에 PDF를 올려주세요.")
        return None

    # PDF 로드 및 분할
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 벡터 DB 생성 (메모리 방식)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = prepare_database()

if retriever:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    template = """당신은 한세대학교 규정 전문가입니다. 아래 문맥을 바탕으로 답변하세요.
    {context}
    질문: {question}
    답변:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

    # 채팅 UI
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
            response = chain.invoke(prompt_input)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
