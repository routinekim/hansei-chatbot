import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch

# --- 1. 보안 설정: Secrets에서 키 불러오기 ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("❌ API 키 설정이 필요합니다. Streamlit Cloud의 Settings > Secrets를 확인하세요.")
    st.stop()

st.set_page_config(page_title="한세대학교 학사 챗봇", page_icon="🎓")

# 메인 타이틀
st.title("🎓 한세대학교 학사 상담 챗봇")
st.markdown("---")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "user_type" not in st.session_state:
    st.session_state.user_type = None

# --- 2. 신분 선택 (질문창 바로 위로 배치) ---
st.subheader("📍 본인의 신분을 먼저 선택해 주세요")
choice = st.radio(
    "정확한 학칙 안내를 위해 필요합니다:",
    ("학부생", "대학원생"),
    horizontal=True, # 가로로 배치해서 공간 절약
    index=0 if st.session_state.user_type is None else ["학부생", "대학원생"].index(st.session_state.user_type)
)

# 신분이 변경되었을 때만 데이터 다시 로드
if choice != st.session_state.user_type:
    with st.spinner(f"🔄 {choice} 학칙 데이터를 분석 중입니다..."):
        target_file = "학부학칙.pdf" if choice == "학부생" else "대학원학칙.pdf"
        
        if os.path.exists(target_file):
            loader = PyPDFLoader(target_file)
            docs = loader.load()
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            db = DocArrayInMemorySearch.from_documents(docs, embeddings)
            st.session_state.retriever = db.as_retriever()
            st.session_state.user_type = choice
            st.toast(f"✅ {choice} 모드로 설정되었습니다.")
        else:
            st.error(f"❌ {target_file} 파일이 서버에 없습니다. 깃허브 업로드 상태를 확인하세요.")
            st.stop()

st.markdown("---")

# --- 3. 대화 인터페이스 ---
# 기존 대화 표시 (스크롤 영역)
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 질문 입력창 (신분 선택 바로 아래에 위치하게 됨)
if prompt := st.chat_input(f"{choice} 관련 궁금한 점을 입력하세요"):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("규정을 확인하고 있습니다..."):
            try:
                # 관련 조항 검색
                relevant_docs = st.session_state.retriever.invoke(prompt)
                context = "\n".join([d.page_content for d in relevant_docs])
                
                # GPT-4o 답변 생성
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                full_prompt = f"당신은 한세대학교 {choice} 전담 상담원입니다. 제공된 학칙을 근거로 답하세요.\n\n[학칙]\n{context}\n\n질문: {prompt}"
                response = llm.invoke(full_prompt)
                
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
