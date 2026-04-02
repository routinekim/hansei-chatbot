import streamlit as st
import os
import io
import sys
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="한세대학교 규정 챗봇", page_icon="🏫")
st.title("🏫 한세대학교 규정 안내 챗봇")
st.markdown("---")

# 2. 보안 설정: Streamlit Secrets에서 API 키 불러오기
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("❌ API 키가 설정되지 않았습니다. Streamlit Cloud의 Settings > Secrets에 OPENAI_API_KEY를 등록해주세요.")
    st.stop()

# 데이터 경로 설정
persist_db = "./db_hansei"

# 3. 데이터베이스 로드 함수 (캐싱 적용으로 속도 향상)
@st.cache_resource
def get_retriever():
    # 배포 환경의 임베딩 설정 (가장 가벼운 모델 사용)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 기존에 생성된 DB 폴더가 있는지 확인
    if os.path.exists(persist_db):
        vectorstore = Chroma(persist_directory=persist_db, embedding_function=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        st.error("❌ 데이터베이스 폴더(db_hansei)를 찾을 수 없습니다. 로컬에서 생성한 폴더를 GitHub에 함께 올려주세요.")
        st.stop()

# 리트리버 및 모델 초기화
try:
    retriever = get_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 프롬프트 템플릿
    template = """당신은 한세대학교 규정 전문가입니다. 
아래 제공된 규정 문맥을 바탕으로 질문에 정확하고 친절하게 답변하세요. 
답변 시 해당 내용이 포함된 '페이지 번호'를 반드시 명시하세요.

문맥: {context}

질문: {question}

답변:"""
    prompt = ChatPromptTemplate.from_template(template)

    # 체인 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    # 4. 채팅 인터페이스 구현
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if user_input := st.chat_input("규정에 대해 궁금한 점을 입력하세요 (예: 교원 징계 시 승진 제한)"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("규정집을 분석하고 있습니다..."):
                try:
                    response = chain.invoke(user_input)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")

except Exception as e:
    st.error(f"시스템 초기화 중 오류 발생: {e}")
