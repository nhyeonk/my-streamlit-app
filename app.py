import os
import tempfile
from operator import itemgetter

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

# 페이지 설정 (반드시 가장 먼저 호출)
st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="📘", layout="wide")

# 벡터 저장소 경로, 컬렉션 이름 (Colab에서 만들 때와 같아야 함)
CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"
LOGO_URL = "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png"
HISTORY_TURNS = 3   # 프롬프트에 넣을 이전 대화 수 (질문+답변 = 1턴)

# 추천 질문 (백서에 답이 있는 질문으로 바꾸면 시연 때 더 좋습니다)
SUGGESTED_QUESTIONS = [
    "미니 프로젝트에 대해 알려주세요.",
    "6개월 과정인데 잘 할 수 있을까요?",
    "비전공자도 따라갈 수 있나요?",
    "에이블스쿨에서는 어떤 것을 배우나요?",
]

# =========================================================
# 시스템 메시지
# =========================================================
S_MSG = """당신은 KT AIVLE School에 관심 있는 예비 지원자와
    교육생을 돕는 친절하고 전문적인 AI 학습 도우미입니다.
 
    [역할]
    1. AIVLE School의 교육과정, 학습 내용, 프로젝트,
      성장 기회 등을 이해하기 쉽게 설명합니다.
    2. 사용자가 AIVLE School에 관심을 갖고
      적극적으로 도전하도록 동기를 부여합니다.
    3. 사용자의 질문 의도를 파악하여 구체적이고
      실질적으로 도움이 되는 답변을 제공합니다.
 
    [답변 원칙]
    1. 반드시 제공된 문서(context)를 근거로 답변합니다.
    2. 문서에 없는 정보는 임의로 만들어내지 않습니다.
    3. 문서에서 확인할 수 없는 내용은
      "제공된 AIVLE School 백서에서 확인되지 않는 내용입니다."
      라고 안내합니다.
    4. 교육과정, 지원 자격, 일정, 혜택 등은
      문서에 명시된 내용만 정확하게 전달합니다.
    5. 전문용어는 초보자도 이해할 수 있도록
      쉬운 설명과 예시를 활용합니다.
 
    [답변 스타일]
    1. 친절하고 긍정적이며 전문적인 말투를 사용합니다.
    2. 핵심 내용을 먼저 설명하고 필요하면 상세히 설명합니다.
    3. 내용이 복잡하면 번호나 목록으로 정리합니다.
    4. 사용자의 관심 분야와 질문에 맞춰
      AIVLE School에서 얻을 수 있는 경험을 소개합니다.
    5. 관련 근거가 있을 때 AIVLE School에서의
       학습과 도전이 어떤 성장으로 이어질 수 있는지 설명합니다.
    6. 과장된 홍보나 근거 없는 취업 보장은 하지 않습니다.
    7. 상황에 적절하다면 관련된 후속 질문을
      한 가지 제안하여 자연스럽게 대화를 이어갑니다.
 
    [문서]
    {context}
    """
# 후속 질문("그거 더 알려줘")을 검색 가능한 질문으로 바꾸는 메시지
REWRITE_MSG = """이전 대화를 참고하여, 사용자의 마지막 질문을 이전 대화 없이도 이해되는 하나의 검색용 질문으로 바꾸세요.
"그거", "더 자세히" 같은 말은 구체적인 대상으로 바꾸세요.
질문만 출력하고 다른 말은 하지 마세요."""


# =========================================================
# 벡터 저장소, 체인
# =========================================================
def format_docs(docs):
    return "\n\n".join(
        f"[p.{doc.metadata.get('page', '?')}]\n{doc.page_content}" for doc in docs
    )


@st.cache_resource
def get_vectorstore():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR,
    )


@st.cache_resource
def get_chain():
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_messages([
        ("system", S_MSG),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.2)
    rag_chain = (
        {
            "context": itemgetter("query") | retriever | format_docs,   # 검색은 재작성된 질문으로
            "input": itemgetter("input"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


@st.cache_resource
def get_rewrite_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_MSG),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return prompt | llm | StrOutputParser()


def make_query(user_input, history):
    if not history:
        return user_input
    return get_rewrite_chain().invoke({"history": history, "input": user_input}).strip() or user_input


# =========================================================
# 문서 업로드 (관리자)
# =========================================================
def add_pdfs(files, reset):
    vectorstore = get_vectorstore()
    if reset:
        vectorstore.reset_collection()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    total = 0
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            docs = PyMuPDF4LLMLoader(tmp_path).load()
            for d in docs:
                d.metadata["source"] = f.name
            chunks = splitter.split_documents(docs)
            if chunks:
                vectorstore.add_documents(chunks)
                total += len(chunks)
        finally:
            os.remove(tmp_path)
    return total


# =========================================================
# 스타일
# =========================================================
st.markdown("""
<style>
.aivle-header {
    background: linear-gradient(135deg, #e6f7f5 0%, #d3efec 100%);
    border: 1px solid #b9e3de;
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 18px;
}
.aivle-header h1 { margin: 0; font-size: 1.8rem; color: #1b3a4b; }
.aivle-header p  { margin: 6px 0 0 0; color: #3c6e71; }
section[data-testid="stSidebar"] .stButton button { justify-content: flex-start; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# 사이드바: 로고 / 관리자 / 추천 질문 / 대화 삭제
# =========================================================
with st.sidebar:
    st.image(LOGO_URL, width=180)
    st.caption("에이블스쿨은 기업 맞춤형 AI/DX 인재를 양성하는 교육 프로그램입니다.")
    st.divider()

    st.subheader("관리자")
    uploaded_files = st.file_uploader("PDF 파일 업로드", type=["pdf"], accept_multiple_files=True)
    reset = st.checkbox("기존 문서를 지우고 새로 저장", help="체크하지 않으면 기존 문서에 추가됩니다.")
    if st.button("📥 문서 반영하기", disabled=not uploaded_files, width="stretch"):
        with st.spinner("문서를 읽고 저장하는 중..."):
            try:
                n = add_pdfs(uploaded_files, reset)
                st.success(f"{n}개 청크를 저장했어요.")
            except Exception as e:
                st.error(f"문서 저장 중 오류: {e}")
    st.caption(f"저장된 청크 수: {get_vectorstore()._collection.count()}")
    st.divider()

    st.subheader("💡 추천 질문")
    for q in SUGGESTED_QUESTIONS:
        if st.button(q, key=f"sq_{q}", width="stretch"):
            st.session_state.pending_question = q
    st.divider()

    if st.button("🗑️ 대화 삭제", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# =========================================================
# 헤더
# =========================================================
st.markdown("""
<div class="aivle-header">
  <h1>📘 AIVLE School 학습 도우미</h1>
  <p>여러분의 도전을 진심으로 응원합니다 🚀</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 채팅
# =========================================================
# 첫 화면 안내 메시지 (대화 기록에는 저장하지 않음)
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "안녕하세요! 에이블스쿨 학습 도우미예요 👋\n\n"
            "교육 과정, 프로젝트, 학습 방법 등 궁금한 점을 편하게 물어보세요. "
            "왼쪽의 **추천 질문**으로 시작해도 좋아요."
        )

# 기존 대화 내용 출력
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# 사용자 입력 처리 (직접 입력 또는 추천 질문 클릭)
typed = st.chat_input("질문을 입력하세요")
user_input = typed or st.session_state.pop("pending_question", None)

if user_input:
    history = st.session_state.messages[-HISTORY_TURNS * 2:]

    # 사용자 메시지 표시
    st.chat_message("user").markdown(user_input)

    # 응답 출력
    with st.chat_message("assistant"):
        try:
            query = make_query(user_input, history)
            response_stream = get_chain().stream(
                {"input": user_input, "query": query, "history": history}
            )
            full_response = st.write_stream(response_stream)
        except Exception as e:
            full_response = f"답변 중 오류가 발생했어요: {e}"
            st.error(full_response)

    # 대화 기록 저장
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=full_response))
