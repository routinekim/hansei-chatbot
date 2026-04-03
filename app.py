import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch

# Streamlit Cloud의 Secrets 또는 로컬의 secrets.toml에서 키를 가져옵니다.
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("API 키 설정이 필요합니다. (Streamlit Secrets 확인)")
    st.stop()

st.set_page_config(page_title="한세대학교 학사 챗봇", page_icon="🎓")

st.title("🎓 한세대학교 학사 상담 챗봇")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "user_type" not in st.session_state:
    st.session_state.user_type = None

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    choice = st.radio("본인의 신분을 선택해 주세요:", ("학부생", "대학원생"))

# 파일 로드 및 인덱싱
if choice and st.session_state.user_type != choice:
    with st.spinner(f"{choice} 데이터를 분석 중입니다..."):
        target_file = "학부학칙.pdf" if choice == "학부생" else "대학원학칙.pdf"
        
        if os.path.exists(target_file):
            loader = PyPDFLoader(target_file)
            docs = loader.load()
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            db = DocArrayInMemorySearch.from_documents(docs, embeddings)
            st.session_state.retriever = db.as_retriever()
            st.session_state.user_type = choice
        else:
            st.error(f"❌ {target_file} 파일이 없습니다.")
            st.stop()

# 대화 창구
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        relevant_docs = st.session_state.retriever.invoke(prompt)
        context = "\n".join([d.page_content for d in relevant_docs])
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        full_prompt = f"한세대학교 {choice} 상담원으로서 다음 내용을 바탕으로 답하세요.\n\n{context}\n\n질문: {prompt}"
        response = llm.invoke(full_prompt)
        
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
