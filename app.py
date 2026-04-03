import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch

# --- 1. 보안 설정 ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("❌ API 키가 설정되지 않았습니다. Secrets를 확인하세요.")
    st.stop()

st.set_page_config(page_title="한세대학교 학사 챗봇", page_icon="🎓", layout="wide")

# --- 2. 사이드바: 사용설명서 및 예시 질문 ---
with st.sidebar:
    st.title("📖 사용설명서")
    st.info("""
    **이 챗봇은 한세대학교 공식 학칙 PDF를 기반으로 답변합니다.**
    
    1. 우측에서 본인의 **신분**을 선택하세요.
    2. 하단 채팅창에 **궁금한 점**을 입력하세요.
    3. 질문이 구체적일수록 정확한 답변이 나옵니다!
    """)
    
    st.divider()
    
    st.subheader("💡 이런 걸 물어보세요!")
    
    # 신분 선택에 따라 다른 예시 질문 보여주기 (세션 상태 활용)
    current_type = st.session_state.get("user_type", "학부생")
    
    if current_type == "학부생":
        st.write("**[학부생 추천 질문]**")
        st.caption("• 졸업 이수 학점과 채플 횟수는?")
        st.caption("• 전과 신청 자격과 시기가 궁금해")
        st.caption("• 성적 장학금 지급 기준이 뭐야?")
        st.caption("• 일반 휴학은 최대 몇 학기까지 가능해?")
    else:
        st.write("**[대학원생 추천 질문]**")
        st.caption("• 석사 학위 논문 제출 자격 시험은?")
        st.caption("• 외국어 시험 면제 기준(토익 등)은?")
        st.caption("• 전공 종합시험 과목 수와 합격 점수는?")
        st.caption("• 수료와 졸업의 차이가 뭐야?")

    st.divider()
    st.caption("© 2026 Hansei Univ. Academic Chatbot")

# --- 3. 메인 화면 ---
st.title("🎓 한세대학교 학사 상담 챗봇")
st.markdown("정확한 학칙 근거를 바탕으로 답변해 드립니다.")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "user_type" not in st.session_state:
    st.session_state.user_type = None

# 신분 선택 레이아웃
st.subheader("📍 본인의 신분을 선택해 주세요")
choice = st.radio(
    "선택한 신분에 맞는 학칙 파일이 로드됩니다:",
    ("학부생", "대학원생"),
    horizontal=True,
    index=0 if st.session_state.user_type is None else ["학부생", "대학원생"].index(st.session_state.user_type)
)

# 데이터 로딩 로직
if choice != st.session_state.user_type:
    with st.spinner(f"🔄 {choice} 데이터를 분석 중입니다..."):
        target_file = "학부학칙.pdf" if choice == "학부생" else "대학원학칙.pdf"
        if os.path.exists(target_file):
            loader = PyPDFLoader(target_file)
            docs = loader.load()
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            db = DocArrayInMemorySearch.from_documents(docs, embeddings)
            st.session_state.retriever = db.as_retriever()
            st.session_state.user_type = choice
            st.rerun() # 사이드바 질문 예시 업데이트를 위해 재실행
        else:
            st.error(f"❌ {target_file} 파일이 없습니다.")
            st.stop()

st.divider()

# 대화 창구 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력
if prompt := st.chat_input(f"{choice} 관련 궁금한 점을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("규정을 검토 중입니다..."):
            try:
                relevant_docs = st.session_state.retriever.invoke(prompt)
                context = "\n".join([d.page_content for d in relevant_docs])
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                full_prompt = f"당신은 한세대학교 {choice} 상담원입니다. 아래 학칙을 바탕으로 답하세요.\n\n{context}\n\n질문: {prompt}"
                response = llm.invoke(full_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
