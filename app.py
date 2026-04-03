import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# 1. 페이지 설정 (최상단)
st.set_page_config(page_title="한세대학교 학사 챗봇", page_icon="🎓", layout="wide")

# 2. 삼성 인터넷 & 모바일 가림 방지 CSS
st.markdown("""
    <style>
    div[data-testid="stChatInputContainer"] {
        position: relative !important; 
        bottom: 0 !important;
        width: 100% !important;
        margin-top: 50px !important;
    }
    div[data-testid="stChatInputContainer"] > div {
        border: 2px solid #007bff !important;
        border-radius: 12px !important;
    }
    .main .block-container {
        padding-bottom: 350px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# API 키 설정
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("API 키 설정이 필요합니다.")
    st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "user_type" not in st.session_state:
    st.session_state.user_type = None

# --- 3. 사이드바: 상세 사용설명서 ---
with st.sidebar:
    st.title("📖 사용설명서")
    st.info("""
    **이 챗봇은 한세대학교 공식 학칙 PDF를 기반으로 답변합니다.**
    1. 본인의 **신분**을 선택하세요.
    2. 화면 맨 아래로 내려가서 **질문**을 입력하세요.
    """)
    
    st.divider()
    
    current_type = st.session_state.get("user_type", "학부생")
    st.subheader(f"💡 {current_type} 추천 질문")
    
    if current_type == "학부생":
        st.caption("• 졸업 이수 학점과 채플 횟수는?\n• 전과 신청 자격과 시기가 궁금해\n• 성적 장학금 지급 기준이 뭐야?")
    else:
        st.caption("• 석사 학위 논문 제출 자격 시험은?\n• 외국어 시험 면제 기준은?\n• 수료와 졸업의 차이가 뭐야?")

    st.divider()
    st.caption("© 2026 Hansei Univ. Chatbot")

# --- 4. 메인 화면 ---
st.title("🎓 한세대학교 학사 상담 챗봇")

# 신분 선택
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
            # 변수 db를 거치지 않고 바로 세션에 저장
            vector_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
            st.session_state.retriever = vector_db.as_retriever()
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

# 질문 입력창
prompt = st.chat_input(f"[{choice}] 여기에 질문을 입력하세요")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("학칙을 확인하고 있습니다..."):
            try:
                # db 대신 세션에 저장된 retriever를 사용합니다.
                relevant_docs = st.session_state.retriever.invoke(prompt)
                context = "\n".join([d.page_content for d in relevant_docs])
                
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                full_prompt = f"당신은 한세대학교 {choice} 상담원입니다. 아래 학칙을 바탕으로 답하세요.\n\n{context}\n\n질문: {prompt}"
                response = llm.invoke(full_prompt)
                
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
