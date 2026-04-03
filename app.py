import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# --- 1. 보안 및 기본 설정 ---
# 반드시 st.set_page_config가 최상단(Import 제외)에 와야 합니다.
st.set_page_config(page_title="한세대학교 학사 챗봇", page_icon="🎓", layout="wide")

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("API 키가 설정되지 않았습니다.")
    st.stop()

# --- 2. 삼성 인터넷 및 모바일 최적화 CSS ---
# 에러가 발생했던 unsafe_allow_set_header를 제거하고 표준 방식으로 작성했습니다.
st.markdown("""
    <style>
    /* 하단 질문창 여백 확보 (삼성 인터넷 가림 방지) */
    .stChatInputContainer {
        padding-bottom: 60px !important;
    }
    /* 메인 컨텐츠 영역 하단 여백 */
    .main .block-container {
        padding-bottom: 100px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 사이드바 (사용설명서) ---
with st.sidebar:
    st.title("📖 사용설명서")
    st.info("신분을 선택하고 궁금한 점을 입력하세요.")
    
    current_type = st.session_state.get("user_type", "학부생")
    st.subheader(f"💡 {current_type} 추천 질문")
    if current_type == "학부생":
        st.caption("• 졸업 이수 학점과 채플 횟수는?\n• 전과 신청 자격과 시기는?\n• 성적 장학금 지급 기준은?")
    else:
        st.caption("• 학위 논문 제출 자격 시험은?\n• 외국어 시험 면제 기준은?\n• 수료와 졸업의 차이는?")

# --- 4. 메인 화면 로직 ---
st.title("🎓 한세대학교 학사 상담 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "user_type" not in st.session_state:
    st.session_state.user_type = None

# 신분 선택
st.subheader("📍 신분 선택")
choice = st.radio(
    "정확한 상담을 위해 선택해 주세요:",
    ("학부생", "대학원생"),
    horizontal=True,
    index=0 if st.session_state.user_type is None else ["학부생", "대학원생"].index(st.session_state.user_type)
)

# 데이터 로딩 (중복 로딩 방지)
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
