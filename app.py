import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. 페이지 설정
st.set_page_config(page_title="한세대학교 규정 챗봇", page_icon="🏫")
st.title("🏫 한세대학교 규정 안내 챗봇")
st.markdown("---")

# 2. 보안 설정 (Streamlit Secrets에서 API 키 로드)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("❌ Secrets에 OPENAI_API_KEY를 등록해주세요.")
    st.stop()

# 3. 데이터베이스 및 리트리버 설정 (캐싱 적용)
@st.cache_resource
def prepare_rag_system():
    pdf_path = "한세대규정_전체.pdf" # 깃허브의 PDF 파일명과 동일해야 함
    
    if not os.path.exists(pdf_path):
        st.error(f"❌ '{pdf_path}' 파일을 찾을 수 없습니다.")
        return None

    # PDF 로드 및 정밀 분할 (노트북LM처럼 맥락을 잘 잡기 위해 overlap 확대)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)

    # 벡터 DB 및 검색기 설정 (k=7로 검색 범위 확장)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 7})

# 시스템 초기화
retriever = prepare_rag_system()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. 대화 맥락 유지를 위한 체인 구성
if retriever:
    # (A) 질문 재구성 프롬프트: 이전 대화를 참고해 질문을 완전한 문장으로 수정
    contextualize_q_system_prompt = (
        "이전 대화 내용과 최신 사용자 질문을 바탕으로, "
        "대화 맥락을 알 수 있는 독립적인 질문으로 다시 작성하세요."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # (B) 답변 생성 프롬프트: 규정 전문가 페르소나 부여
    system_prompt = (
        "당신은 한세대학교 규정 전문가입니다. "
        "아래의 문맥(context)을 사용하여 질문에 답변하세요. "
        "답변 시 반드시 해당 조항과 페이지를 명시하세요. "
        "민감한 정보(위원 명단 등)에 대해서는 규정에 명시된 공개/비공개 원칙을 철저히 따르세요."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # (C) 전체 RAG 체인 통합
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 5. 채팅 UI 및 세션 관리
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 기존 대화 표시
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # 질문 입력 및 처리
    if user_input := st.chat_input("규정에 대해 질문하세요..."):
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("규정집 분석 중..."):
                result = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                answer = result["answer"]
                st.markdown(answer)
                
                # 대화 기록 저장 (맥락 유지의 핵심)
                st.session_state.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=answer),
                ])
