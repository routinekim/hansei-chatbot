import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# --- 1. 보안 및 기본 설정 ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("API 키가 설정되지 않았습니다.")
    st.stop()

# 모바일 대응을 위해 레이아웃을 'centered'로 하거나 여백 최적화
st.set_page_config(page_title="한세대학교 학사 챗봇", page_icon="🎓", layout="wide")

# 모바일 하단 가림 방지를 위한 커스텀 CSS 추가
st.markdown("""
    <style>
    /* 하단 질문창이 브라우저 도구바에 가려지지 않도록 여백 추가 */
    .stChatInputContainer {
        padding-bottom: 50px !important;
    }
    /* 모바일에서 사이드바가 너무 넓게 차지하지 않도록 조정 */
    section[data-testid="stSidebar"] {
        width: 250px !important;
    }
    </style>
    """, unsafe_allow_set_header=False, unsafe_allow_html=True)

# --- 2. 사이드바 (내용 동일) ---
with st.sidebar:
    st.title("📖 사용설명서")
    st.info("본인의 신분을 선택한 후 질문을 입력하세요.")
    
    current_type = st.session_state.get("user_type", "학부생")
    st.subheader(f"💡 {current_type} 추천 질문")
    if current_type == "학부생":
        st.caption("• 졸업 이수 학점과 채플 횟수는?\n• 전과 신청 자격과 시기는?\n• 성적 장학금 지급 기준은?")
    else:
        st.caption("• 학위 논문 제출 자격 시험은?\n• 외국어 시험 면제 기준은?\n• 수료와 졸업의 차이는?")

# --- 3. 메인 화면 ---
st.title("🎓 한세대학교 학사 상담 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "user_type" not in st.session_state:
    st.session_state.user_type = None

# 신분 선택 (가시성 확보)
st.subheader("📍 신분 선택")
choice = st.radio(
    "정확한 상담을 위해 선택해 주세요:",
    ("학부생", "대학원생"),
    horizontal=True,
    index=0 if st.session_state.user_type is None else ["학부생", "대학원생"].index(st.session_state.user_type)
)

# 데이터 로딩
if choice != st.session_state.user_type:
    with st.spinner(f"🔄 {choice} 데이터 로딩 중..."):
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

# --- [핵심] 질문 입력 ---
# 삼성 인터넷의 하단 바 문제를 피하기 위해 안내 문구를 더 명확히 함
prompt = st.chat_input(f"[{choice}] 궁금한 점을 입력하세요")

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
                full_prompt = f"당신은 한세대학교 {choice} 상담원입니다. 학칙에 근거하여 답하세요.\n\n{context}\n\n질문: {prompt}"
                response = llm.invoke(full_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"오류: {e}")

# 마지막에 빈 공간을 추가하여 스크롤 여유 확보
st.write("<br><br><br>", unsafe_allow_html=True)
