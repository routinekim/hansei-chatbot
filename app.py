import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# 1. 페이지 설정 (최상단 배치)
st.set_page_config(page_title="한세대학교 학사 챗봇", page_icon="🎓", layout="wide")

# 2. 삼성 인터넷 및 모바일 브라우저 최적화 CSS
st.markdown("""
    <style>
    /* 질문창이 본문 흐름을 따라가게 하여 삼성 인터넷 가림 방지 */
    div[data-testid="stChatInputContainer"] {
        position: relative !important; 
        bottom: 0 !important;
        width: 100% !important;
        margin-top: 50px !important;
    }
    
    /* 입력창 테두리 강조 */
    div[data-testid="stChatInputContainer"] > div {
        border: 2px solid #007bff !important;
        border-radius: 12px !important;
    }

    /* 하단에 거대한 여백을 주어 툴바 위로 확실히 노출 */
    .main .block-container {
        padding-bottom: 350px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# API 키 설정
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("API 키 설정이 필요합니다. Streamlit Secrets를 확인하세요.")
    st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "user_type" not in st.session_state:
    st.session_state.user_type = None

# --- 3. 사이드바: 상세 사용설명서 복구 ---
with st.sidebar:
    st.title("📖 사용설명서")
    st.info("""
    **이 챗봇은 한세대학교 공식 학칙 PDF를 기반으로 답변합니다.**
    
    1. 우측 메인 화면에서 본인의 **신분**을 선택하세요.
    2. 화면 맨 아래로 내려가서 **궁금한 점**을 입력하세요.
    3. 질문이 구체적일수록 정확한 답변이 나옵니다!
    """)
    
    st.divider()
    
    # 현재 선택된 신분에 따라 변하는 예시 질문
    current_type = st.session_state.get("user_type", "학부생")
    st.subheader(f"💡 {current_type} 추천 질문")
    
    if current_type == "학부생":
        st.write("**[학부생 베스트 질문]**")
        st.caption("• 졸업 이수 학점과 채플 횟수는?")
        st.caption("• 다른 학과로 전과하는 방법과 시기는?")
        st.caption("• 성적 장학금을 받기 위한 최소 기준은?")
        st.caption("• 일반 휴학은 최대 몇 학기까지 가능해?")
        st.caption("• 재수강 시 성적 상한선이 있어?")
    else:
        st.write("**[대학원생 베스트 질문]**")
        st.caption("• 석사 학위 논문 제출 자격 시험은?")
        st.caption("• 외국어 시험 면제 기준(토익 등)은?")
        st.caption("• 전공 종합시험 과목 수와 합격 점수는?")
        st.caption("• 수료와 졸업의 차이가 뭐야?")
        st.caption("• 타 전공 수업도 이수 학점으로 인정돼?")

    st.divider()
    st.caption("© 2026 Hansei Univ. Academic Chatbot")

# --- 4. 메인 화면 로직 ---
st.title("🎓 한세대학교 학사 상담 챗봇")
st.markdown("정확한 학칙 근거를 바탕으로 답변해 드립니다.")

# 신분 선택 레이아웃
st.subheader("📍 본인의 신분을 선택해 주세요")
choice = st.radio(
    "선택한 신분에 맞는 학칙 파일이 로드됩니다:",
    ("학부생", "대학원생"),
    horizontal=True,
    index=0 if st.session_state.user_type is None else ["학부생", "대학원생"].index(st.session_state.user_type)
)

# 데이터 로딩 (신분 변경 시 또는 최초 실행 시)
if choice != st.session_state.user_type or st.session_state.retriever is None:
    with st.spinner(f"🔄 {choice} 데이터를 분석 중입니다..."):
        target_file = "학부학칙.pdf" if choice == "학부생" else "대학원학칙.pdf"
        if os.path.exists(target_file):
            loader = PyPDFLoader(target_file)
            docs = loader.load()
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            # NameError 방지를 위해 세션에 즉시 저장
            temp_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
            st.session_state.retriever = temp_db.as_retriever()
            st.session_state.user_type = choice
            st.rerun() 
        else:
            st.error(f"❌ {target_file} 파일이 없습니다.")
            st.stop()

st.divider()

# 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력창 (삼성 인터넷 대응 구조)
prompt = st.chat_input(f"[{choice}] 여기에 질문을 입력하세요 (안 보이면 아래로 스크롤)")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("규정을 검토 중입니다..."):
            try:
                # 세션에 저장된 retriever를 안전하게 사용
                relevant_docs = st.session_state.retriever.invoke(prompt)
                context = "\n".join([d.page_content for d in relevant_docs])
                llm = ChatOpenAI(
