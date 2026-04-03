import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# 1. 페이지 설정 (가장 먼저 실행)
st.set_page_config(page_title="한세대학교 학사 챗봇", page_icon="🎓", layout="wide")

# 2. 삼성 인터넷 하단바 가림 방지 전용 CSS
# 하단 여백을 100px 이상으로 대폭 늘려 입력창을 위로 밀어 올립니다.
st.markdown("""
    <style>
    /* 채팅 입력창 컨테이너 위치 강제 조정 */
    .stChatInputContainer {
        bottom: 70px !important; 
        background-color: rgba(255, 255, 255, 0.9) !important;
        padding: 10px !important;
    }
    
    /* 전체 화면 하단에 큰 여백을 주어 스크롤이 끝까지 내려가게 함 */
    .main .block-container {
        padding-bottom: 180px !important;
    }

    /* 모바일에서 입력창이 가려지는 것을 방지하기 위한 추가 설정 */
    @media screen and (max-width: 768px) {
        .stChatInputContainer {
            bottom: 80px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# API 키 및 환경 설정
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("API 키 설정이 필요합니다.")
    st.stop()

# --- 이하 로직 동일 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "user_type" not in st.session_state:
    st.session_state.user_type = None

with st.sidebar:
    st.title("📖 사용설명서")
    st.info("신분을 선택하고 질문을 입력하세요.")
    current_type = st.session_state.get("user_type", "학부생")
    st.subheader(f"💡 {current_type} 추천 질문")
    if current_type == "학부생":
        st.caption("• 졸업 이수 학점과 채플 횟수는?\n• 전과 신청 자격과 시기는?")
    else:
        st.caption("• 학위 논문 제출 자격 시험은?\n• 수료와 졸업의 차이는?")

st.title("🎓 한세대학교 학사 상담 챗봇")

st.subheader("📍 신분 선택")
choice = st.radio(
    "정확한 상담을 위해 선택해 주세요:",
    ("학부생", "대학원생"),
    horizontal=True,
    index=0 if st.session_state.user_type is None else ["학부생", "대학원생"].index(st.session_state.user_type)
)

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력
prompt = st.chat_input(f"[{choice}] 질문을 입력하세요")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("규정을 확인 중..."):
            try:
                relevant_docs = st.session_state.retriever.invoke(prompt)
                context = "\n".join([d.page_content for d in relevant_docs])
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                full_prompt = f"한세대학교 {choice} 상담원입니다. 학칙에 근거하여 답하세요.\n\n{context}\n\n질문: {prompt}"
                response = llm.invoke(full_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"오류: {e}")
