import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# 1. 페이지 설정 (아이콘 및 타이틀)
st.set_page_config(page_title="한세대학교 챗봇", page_icon="🎓", layout="wide")

# 2. 커스텀 CSS 주입 (제공해주신 style.css 내용 기반)
st.markdown(f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.8/dist/web/static/pretendard.css" />
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        /* 기본 Streamlit 요소 숨기기 */
        #MainMenu, footer, header {{visibility: hidden;}}
        .stDeployButton {{display:none;}}
        
        /* 사용자 정의 변수 */
        :root {{
            --primary: #04447c;
            --secondary: #203546;
            --bg-color: #ffffff;
            --border: #e2e8f0;
            --text-main: #333333;
        }}

        /* 전체 배경 및 폰트 */
        .main {{
            background-color: var(--bg-color);
            font-family: 'Pretendard', sans-serif;
        }}

        /* 헤더 스타일 */
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

        /* 말풍선 스타일 수정 */
        .stChatMessage {{
            background-color: transparent !important;
            border: none !important;
        }}
        
        /* 봇 말풍선 (제공 디자인 적용) */
        [data-testid="stChatMessage"]:nth-child(even) .stChatMessageContent {{
            background-color: #EBF2FA !important;
            color: var(--text-main) !important;
            border-radius: 16px !important;
            border-top-left-radius: 4px !important;
            padding: 12px 16px !important;
        }}

        /* 사용자 말풍선 (제공 디자인 적용) */
        [data-testid="stChatMessage"]:nth-child(odd) .stChatMessageContent {{
            background-color: var(--primary) !important;
            color: white !important;
            border-radius: 16px !important;
            border-top-right-radius: 4px !important;
            padding: 12px 16px !important;
        }}

        /* 질문 입력창 (삼성 인터넷 대응 및 디자인 통합) */
        div[data-testid="stChatInputContainer"] {{
            padding: 10px !important;
            background-color: white !important;
            border-top: 1px solid var(--border) !important;
        }}
    </style>
    
    <div class="custom-header">🎓 Hansei 챗봇</div>
    <div style="margin-top: 70px;"></div>
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

# --- 데이터 로딩 및 사이드바 (디자인 유지) ---
with st.sidebar:
    st.title("📖 사용설명서")
    choice = st.radio("신분을 선택하세요:", ("학부생", "대학원생"), horizontal=True)
    st.info("💡 질문이 구체적일수록 답변이 정확합니다.")

# 신분 변경 시 데이터 로드
if st.session_state.get("user_type") != choice:
    with st.spinner("데이터 분석 중..."):
        target = "학부학칙.pdf" if choice == "학부생" else "대학원학칙.pdf"
        if os.path.exists(target):
            loader = PyPDFLoader(target)
            db = DocArrayInMemorySearch.from_documents(loader.load(), OpenAIEmbeddings(model="text-embedding-3-small"))
            st.session_state.retriever = db.as_retriever()
            st.session_state.user_type = choice
            st.rerun()

# --- 대화창 인터페이스 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력해주세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        relevant_docs = st.session_state.retriever.invoke(prompt)
        context = "\n".join([d.page_content for d in relevant_docs])
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm.invoke(f"한세대학교 {choice} 상담원입니다. 학칙 근거 답변:\n{context}\n질문:{prompt}")
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
