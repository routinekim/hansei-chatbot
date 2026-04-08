import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# 1. 페이지 설정
st.set_page_config(page_title="한세대학교 챗봇", page_icon="🎓", layout="wide")

# 2. 통합 디자인 CSS (제공해주신 style.css 핵심 반영)
st.markdown(f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.8/dist/web/static/pretendard.css" />
    <style>
        #MainMenu, footer, header {{visibility: hidden;}}
        .stDeployButton {{display:none;}}
        
        :root {{
            --primary: #04447c;
            --secondary: #203546;
            --bg-color: #ffffff;
            --border: #e2e8f0;
            --text-main: #333333;
            --bubble-bot: #EBF2FA;
        }}

        .main {{ background-color: var(--bg-color); font-family: 'Pretendard', sans-serif; }}

        /* 상단 고정 헤더 */
        .custom-header {{
            background-color: var(--primary);
            color: white;
            padding: 15px 20px;
            text-align: center;
            font-weight: 600;
            font-size: 1.2rem;
            position: fixed;
            top: 0; left: 0; right: 0;
            z-index: 99;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        /* 말풍선 디자인 커스텀 */
        [data-testid="stChatMessage"]:nth-child(even) .stChatMessageContent {{
            background-color: var(--bubble-bot) !important;
            color: var(--text-main) !important;
            border-radius: 16px !important;
            border-top-left-radius: 4px !important;
        }}

        [data-testid="stChatMessage"]:nth-child(odd) .stChatMessageContent {{
            background-color: var(--primary) !important;
            color: white !important;
            border-radius: 16px !important;
            border-top-right-radius: 4px !important;
        }}

        /* 질문 입력창 (삼성 인터넷 대응을 위해 위치 조정) */
        div[data-testid="stChatInputContainer"] {{
            position: relative !important;
            bottom: 0 !important;
            margin-top: 30px !important;
            padding: 10px !important;
        }}
        
        .main .block-container {{ padding-bottom: 300px !important; }}
    </style>
    
    <div class="custom-header">🎓 Hansei 챗봇</div>
    <div style="margin-top: 80px;"></div>
    """, unsafe_allow_html=True)

# API 키 설정
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("API 키가 없습니다.")
    st.stop()

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# 사이드바 설정
with st.sidebar:
    st.title("📖 사용설명서")
    choice = st.radio("신분을 선택하세요:", ("학부생", "대학원생"), horizontal=True)
    st.divider()
    st.info("화면 하단의 메뉴를 클릭하거나 직접 질문을 입력하세요.")

# 데이터 로드 로직
if st.session_state.get("user_type") != choice or st.session_state.retriever is None:
    with st.spinner("학칙 분석 중..."):
        target = "학부학칙.pdf" if choice == "학부생" else "대학원학칙.pdf"
        if os.path.exists(target):
            loader = PyPDFLoader(target)
            db = DocArrayInMemorySearch.from_documents(loader.load(), OpenAIEmbeddings(model="text-embedding-3-small"))
            st.session_state.retriever = db.as_retriever()
            st.session_state.user_type = choice
            st.rerun()

# 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 🎯 6개 퀵 메뉴 (자주하는 질문) 구현 ---
st.markdown("### 💡 자주하는 질문")
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

quick_questions = [
    ("📅 학사일정", "올해 주요 학사일정을 알려줘"),
    ("💰 장학금", "성적 장학금 지급 기준이 뭐야?"),
    ("📄 증명서", "재학증명서나 성적증명서는 어디서 발급받아?"),
    ("📚 학사정보", "졸업을 위해 이수해야 하는 필수 학점은?"),
    ("🌐 인터넷", "교내 와이파이(Wi-Fi) 연결 방법을 알려줘"),
    ("📞 전화번호", "학사지원팀 등 주요 부서 전화번호를 알려줘")
]

# 버튼 클릭 시 질문 처리 함수
def handle_quick_click(q_text):
    st.session_state.messages.append({"role": "user", "content": q_text})
    st.rerun()

# 6개 버튼 배치
cols = [col1, col2, col3, col4, col5, col6]
for idx, (label, query) in enumerate(quick_questions):
    if cols[idx].button(label, use_container_width=True):
        handle_quick_click(query)

st.divider()

# 직접 입력창
if prompt := st.chat_input("질문을 입력해주세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# AI 답변 생성 로직
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("규정을 확인 중입니다..."):
            try:
                relevant_docs = st.session_state.retriever.invoke(user_query)
                context = "\n".join([d.page_content for d in relevant_docs])
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                response = llm.invoke(f"한세대학교 {choice} 상담원입니다. 학칙 근거 답변:\n{context}\n질문:{user_query}")
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                st.rerun()
            except Exception as e:
                st.error(f"오류: {e}")
