import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# 1. 페이지 설정
st.set_page_config(page_title="한세대학교 학사 챗봇", page_icon="🎓", layout="wide")

# 2. 삼성 인터넷 & 모바일 브라우저 강제 레이아웃 수정
st.markdown("""
    <style>
    /* 1. 전체 컨텐츠 하단에 거대한 여백 생성 (질문창이 위로 밀려 올라감) */
    .main .block-container {
        padding-bottom: 200px !important;
    }

    /* 2. 하단 고정 입력창의 위치를 위로 강제 이동 */
    div[data-testid="stChatInputContainer"] {
        bottom: 100px !important; /* 삼성 인터넷 하단바가 보통 60~80px입니다 */
        position: fixed;
        background-color: white !important;
        z-index: 999;
    }

    /* 3. 입력창 테두리 강조 (안 보일 때를 대비해 눈에 띄게 설정) */
    div[data-testid="stChatInputContainer"] > div {
        border: 2px solid #ff4b4b !important;
        border-radius: 10px;
    }

    /* 4. 모바일 전용 추가 여백 */
    @media screen and (max-width: 768px) {
        div[data-testid="stChatInputContainer"] {
            bottom: 120px !important; 
        }
    }
    </style>
    """, unsafe_allow_html=True)

# API 키 설정
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("API 키 설정이 필요합니다.")
    st.stop()

# --- 세션 및 데이터 로직 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "user_type" not in st.session_state:
    st.session_state.user_type = None

# 사이드바 사용설명서
with st.sidebar:
    st.title("📖 사용설명서")
    st.info("삼성 인터넷에서 질문창이 안 보일 경우 화면을 위로 살짝 스크롤해 보세요.")
    current_type = st.session_state.get("user_type", "학부생")
    st.subheader(f"💡 {current_type} 추천 질문")
    if current_type == "학부생":
        st.caption("• 졸업 요건과 채플 횟수\n• 전과 및 장학금 기준")
    else:
        st.caption("• 논문 제출 자격 시험\n• 외국어 시험 면제 기준")

st.title("🎓 한세대학교 학사 상담 챗봇")

# 신분 선택
st.subheader("📍 신분 선택")
choice = st.radio(
    "정확한 상담을 위해 선택해 주세요:",
    ("학부생", "대학원생"),
    horizontal=True,
    index=0 if st.session_state.user_type is None else ["학부생", "대학원생"].index(st.session_state.user_type)
)

# 데이터 로딩
if choice != st.session_state.user_type:
    with st.spinner(f"🔄 {choice} 데이터 로드 중..."):
        target_file = "학부학칙.pdf" if choice == "학부생" else "대학원학칙.pdf"
        if os.path.exists(target_file):
            loader = PyPDFLoader(target_file)
            docs = loader.load()
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            db = DocArrayInMemorySearch.from_documents(docs, embeddings)
            st.session_state.retriever = db.as_retriever()
            st.session_state.user_type = choice
            st.rerun()

st.divider()

# 대화 내용
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력 (강력한 위치 조정 적용 대상)
prompt = st.chat_input(f"[{choice}] 여기에 질문을 입력하세요")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("학칙 검토 중..."):
            try:
                relevant_docs = st.session_state.retriever.invoke(prompt)
                context = "\n".join([d.page_content for d in relevant_docs])
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                full_prompt = f"한세대학교 {choice} 상담원입니다. 학칙에 근거하여 답변하세요.\n\n{context}\n\n질문: {prompt}"
                response = llm.invoke(full_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"오류: {e}")
