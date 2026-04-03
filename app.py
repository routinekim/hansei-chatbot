import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# 1. 페이지 설정
st.set_page_config(page_title="한세대학교 학사 챗봇", page_icon="🎓", layout="wide")

# 2. 삼성 인터넷 대응 커스텀 스타일 (고정 위치 해제 및 강제 노출)
st.markdown("""
    <style>
    /* 하단 고정 바를 해제하고 본문 흐름에 포함 */
    div[data-testid="stChatInputContainer"] {
        position: relative !important; 
        bottom: 0 !important;
        width: 100% !important;
        padding: 20px 0 !important;
        margin-top: 30px !important;
    }
    
    /* 입력창 테두리를 파란색으로 강조하여 눈에 띄게 함 */
    div[data-testid="stChatInputContainer"] > div {
        border: 3px solid #007bff !important;
        border-radius: 15px !important;
    }

    /* 전체 컨텐츠 하단에 아주 큰 여백을 주어 툴바 위로 올림 */
    .main .block-container {
        padding-bottom: 300px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# API 키 설정
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("API 키 설정이 필요합니다.")
    st.stop()

# --- 세션 상태 및 로직 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "user_type" not in st.session_state:
    st.session_state.user_type = None

# 사이드바 사용설명서
with st.sidebar:
    st.title("📖 사용설명서")
    st.warning("⚠️ 삼성 인터넷 사용자 필독\n질문창이 안 보이면 화면을 끝까지 아래로 내려보세요.")
    current_type = st.session_state.get("user_type", "학부생")
    if current_type == "학부생":
        st.caption("• 졸업 요건과 채플 횟수\n• 전과 및 장학금 기준")
    else:
        st.caption("• 논문 제출 자격 시험\n• 수료와 졸업의 차이")

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

# 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- [핵심] 질문 입력창 ---
# 이제 입력창이 하단에 고정되지 않고, 대화가 길어지면 아래로 같이 내려갑니다.
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
