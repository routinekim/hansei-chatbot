import os
import sys
import warnings

# 1. API 키 설정
os.environ["OPENAI_API_KEY"] = "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"

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.vectorstores import DocArrayInMemorySearch
except ImportError:
    print("❌ 라이브러리 누락! 터미널 실행: pip install langchain-openai langchain-community pypdf docarray")
    sys.exit()

warnings.filterwarnings("ignore")

def start_bot():
    try:
        print("\n" + "="*50)
        print("🎓 한세대학교 학사 행정 지능형 챗봇")
        print("="*50)
        
        # [신규] 사용자 유형 선택 단계
        print("\n안녕하세요! 본인의 신분을 선택해 주세요.")
        print("1. 학부생 (학부 학칙 기준)")
        print("2. 대학원생 (대학원 학칙 기준)")
        
        choice = input("\n번호를 입력하세요 (1 또는 2): ").strip()
        
        target_file = ""
        user_type = ""
        
        if choice == '1':
            target_file = "학부학칙.pdf"
            user_type = "학부생"
        elif choice == '2':
            target_file = "대학원학칙.pdf"
            user_type = "대학원생"
        else:
            print("⚠️ 잘못된 입력입니다. 프로그램을 재시작해 주세요.")
            return

        if not os.path.exists(target_file):
            print(f"❌ {target_file} 파일이 폴더에 없습니다. 파일명을 확인해 주세요.")
            return

        print(f"\n🚀 {user_type} 맞춤형 학칙 데이터를 로딩 중입니다...")
        loader = PyPDFLoader(target_file)
        docs = loader.load()
        
        # 검색 엔진 구축
        db = DocArrayInMemorySearch.from_documents(
            docs, 
            OpenAIEmbeddings(model="text-embedding-3-small")
        )
        retriever = db.as_retriever()
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        print("\n" + "*"*45)
        print(f"✅ {user_type}님, 상담 준비가 완료되었습니다!")
        print("   궁금한 점을 물어보세요 (종료: q)")
        print("*"*45)
        
        while True:
            q = input("\n질문: ")
            if q.lower() == 'q': break
            
            print("🔍 관련 학칙 검토 중...", end="\r")
            
            # 검색 및 답변 생성
            relevant_docs = retriever.invoke(q)
            context = "\n".join([d.page_content for d in relevant_docs])
            
            # 프롬프트에 사용자 유형 반영
            prompt = f"당신은 한세대학교 {user_type} 전담 상담원입니다. 다음 학칙을 바탕으로 답변하세요.\n\n내용: {context}\n\n질문: {q}"
            response = llm.invoke(prompt)
            
            print(f"\n[AI 상담원]:\n{response.content}")

    except Exception as e:
        print(f"\n❌ 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    start_bot()
