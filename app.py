import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile
import os

# =====================
# 페이지 설정
# =====================
st.set_page_config(
    page_title="PDF RAG 챗봇",
    page_icon="📚",
    layout="wide"
)

# =====================
# 스타일링
# =====================
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #4A90A4;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# 헤더
# =====================
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📚 PDF RAG 챗봇")
st.caption("Powered by Gemini 2.5 Flash + LangChain")
st.markdown('</div>', unsafe_allow_html=True)

# =====================
# API Key 설정
# =====================
try:
    GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("⚠️ `GEMINI_API_KEY`가 설정되지 않았습니다. Streamlit Secrets에 추가해주세요.")
    st.stop()

# =====================
# 세션 상태 초기화
# =====================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# =====================
# PDF 처리 함수
# =====================
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

def process_pdf(pdf_path: str):
    """PDF를 로드하고 벡터 스토어 생성"""
    
    # PDF 로드
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    
    # 임베딩 및 벡터 스토어 생성
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore, len(splits)

def create_chain(vectorstore):
    """RAG 체인 생성"""
    
    # LLM 초기화 (Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    # 메모리 설정
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Conversational Retrieval Chain 생성
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    
    return chain

# =====================
# 사이드바 - PDF 업로드
# =====================
with st.sidebar:
    st.header("📄 PDF 업로드")
    
    uploaded_file = st.file_uploader(
        "PDF 파일을 업로드하세요",
        type=["pdf"],
        help="업로드한 PDF 문서를 기반으로 질문에 답변합니다."
    )
    
    # 기본 test.pdf 사용 옵션
    use_default = st.checkbox(
        "기본 test.pdf 사용",
        help="저장소에 있는 test.pdf 파일을 사용합니다."
    )
    
    process_btn = st.button("🚀 PDF 처리 시작", type="primary", use_container_width=True)
    
    st.divider()
    
    # 상태 표시
    if st.session_state.vectorstore is not None:
        st.success("✅ PDF 처리 완료!")
        st.info("💬 채팅창에서 질문하세요.")
    
    # 초기화 버튼
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.chain = None
        st.rerun()

# =====================
# PDF 처리 로직
# =====================
if process_btn:
    pdf_path = None
    
    # 업로드된 파일 처리
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
    # 기본 파일 사용
    elif use_default and os.path.exists("test.pdf"):
        pdf_path = "test.pdf"
    else:
        st.sidebar.error("⚠️ PDF 파일을 업로드하거나 기본 파일을 선택하세요.")
    
    if pdf_path:
        with st.spinner("📖 PDF를 분석하고 있습니다..."):
            try:
                vectorstore, num_chunks = process_pdf(pdf_path)
                st.session_state.vectorstore = vectorstore
                st.session_state.chain = create_chain(vectorstore)
                st.session_state.messages = []
                st.session_state.chat_history = []
                
                st.sidebar.success(f"✅ 처리 완료! ({num_chunks}개 청크 생성)")
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"❌ 오류 발생: {str(e)}")
            finally:
                # 임시 파일 정리
                if uploaded_file is not None and pdf_path:
                    os.unlink(pdf_path)

# =====================
# 채팅 인터페이스
# =====================
if st.session_state.vectorstore is None:
    st.info("👈 사이드바에서 PDF 파일을 업로드하고 처리를 시작하세요.")
else:
    # 이전 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📎 참조 문서"):
                    st.markdown(message["sources"])

    # 사용자 입력
    if prompt := st.chat_input("PDF 내용에 대해 질문하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("🤔 답변을 생성하고 있습니다..."):
                try:
                    # RAG 체인 실행
                    response = st.session_state.chain({
                        "question": prompt
                    })
                    
                    answer = response["answer"]
                    source_docs = response.get("source_documents", [])
                    
                    # 응답 표시
                    st.markdown(answer)
                    
                    # 참조 문서 표시
                    sources_text = ""
                    if source_docs:
                        with st.expander("📎 참조 문서"):
                            for i, doc in enumerate(source_docs, 1):
                                page_num = doc.metadata.get("page", "N/A")
                                content_preview = doc.page_content[:200] + "..."
                                source_info = f"**[{i}] 페이지 {page_num}**\n\n{content_preview}\n\n---\n"
                                st.markdown(source_info)
                                sources_text += source_info
                    
                    # 메시지 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_text
                    })
                    
                except Exception as e:
                    error_msg = f"❌ 오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# =====================
# 푸터
# =====================
st.divider()
st.caption("Made with ❤️ using Streamlit, LangChain & Gemini 2.5 Flash")
