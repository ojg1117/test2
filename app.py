import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile
import os

# =====================
# 페이지 설정
# =====================
st.set_page_config(
    page_title="오잭형이 만든 PDF RAG 챗봇",
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
</style>
""", unsafe_allow_html=True)

# =====================
# 헤더
# =====================
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📚 오잭형이 만든 PDF RAG 챗봇")
st.caption("Powered by Gemini 2.5 Flash + LangChain LCEL")
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
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================
# LLM 및 임베딩 초기화
# =====================
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

# =====================
# PDF 처리 함수
# =====================
def process_pdf(pdf_path: str):
    """PDF를 로드하고 벡터 스토어 생성"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore, len(splits)

# =====================
# RAG 체인 함수 (LCEL)
# =====================
def get_rag_response(query: str, vectorstore, chat_history: list):
    """LCEL 기반 RAG 응답 생성"""
    
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 관련 문서 검색
    retrieved_docs = retriever.invoke(query)
    
    # 컨텍스트 구성
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 PDF 문서 기반 질의응답 전문가입니다.
주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 친절하게 답변하세요.
컨텍스트에 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.

[컨텍스트]
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # LCEL 체인 구성
    chain = prompt | llm | StrOutputParser()
    
    # 체인 실행
    response = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": query
    })
    
    return response, retrieved_docs

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
    
    use_default = st.checkbox(
        "깃허브에 업로도된 test.pdf 사용",
        help="깃허브 저장소에 있는 test.pdf 파일을 사용합니다."
    )
    
    process_btn = st.button("🚀 PDF 처리 시작", type="primary", use_container_width=True)
    
    st.divider()
    
    if st.session_state.vectorstore is not None:
        st.success("✅ PDF 처리 완료!")
        st.info("💬 채팅창에서 질문하세요.")
    
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# =====================
# PDF 처리 로직
# =====================
if process_btn:
    pdf_path = None
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
    elif use_default and os.path.exists("test.pdf"):
        pdf_path = "test.pdf"
    else:
        st.sidebar.error("⚠️ PDF 파일을 업로드하거나 기본 파일을 선택하세요.")
    
    if pdf_path:
        with st.spinner("📖 PDF를 분석하고 있습니다..."):
            try:
                vectorstore, num_chunks = process_pdf(pdf_path)
                st.session_state.vectorstore = vectorstore
                st.session_state.messages = []
                st.session_state.chat_history = []
                
                st.sidebar.success(f"✅ 처리 완료! ({num_chunks}개 청크 생성)")
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"❌ 오류 발생: {str(e)}")
            finally:
                if uploaded_file is not None and pdf_path and os.path.exists(pdf_path):
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
            if "sources" in message and message["sources"]:
                with st.expander("📎 참조 문서"):
                    st.markdown(message["sources"])

    # 사용자 입력
    if prompt := st.chat_input("PDF 내용에 대해 질문하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 답변을 생성하고 있습니다..."):
                try:
                    answer, source_docs = get_rag_response(
                        prompt,
                        st.session_state.vectorstore,
                        st.session_state.chat_history
                    )
                    
                    st.markdown(answer)
                    
                    # 참조 문서 표시
                    sources_text = ""
                    if source_docs:
                        with st.expander("📎 참조 문서"):
                            for i, doc in enumerate(source_docs, 1):
                                page_num = doc.metadata.get("page", "N/A")
                                content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                source_info = f"**[{i}] 페이지 {page_num}**\n\n{content_preview}\n\n---\n"
                                st.markdown(source_info)
                                sources_text += source_info
                    
                    # 대화 기록 업데이트
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    
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
                        "content": error_msg,
                        "sources": ""
                    })

# =====================
# 푸터
# =====================
st.divider()
st.caption("Made with ❤️ using Streamlit, LangChain LCEL & Gemini 2.5 Flash")
