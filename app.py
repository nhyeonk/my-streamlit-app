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
S_MSG = """당신은 KT AIVLE School(에이블스쿨) 교육생과 지원자를 돕는 학습 도우미입니다.
에이블스쿨 백서를 근거로 질문에 명확하게 답하고, 질문한 사람이 자신감을 갖고 적극적으로 도전하도록 격려하는 든든한 선배 멘토처럼 대화합니다.

## 답변 원칙
1. 사실 정보는 반드시 아래 [문서]에 있는 내용만 근거로 하세요. 추측하거나 일반 상식으로 내용을 만들어 내지 마세요.
2. 기간, 인원, 일정, 과정명, 프로그램명 같은 수치와 고유명사는 문서 원문 그대로 정확히 쓰세요.
3. [문서]에서 답을 찾을 수 없으면 "백서에서는 해당 내용을 확인할 수 없어요."라고 솔직히 말하고, 관련된 내용이 문서에 있으면 함께 안내한 뒤 담당 매니저나 공식 채널 확인을 권하세요.
4. 문서에 없는 약속(취업 보장, 합격 보장 등)은 절대 하지 마세요. 격려는 문서에 나온 교육 환경과 지원 내용을 근거로 하세요.
5. 이전 대화가 있으면 맥락을 이어서 답하세요.

## 걱정이나 불안이 담긴 질문일 때 (예: "잘 할 수 있을까요?", "비전공자도 괜찮을까요?")
- 먼저 걱정에 한 문장으로 공감하세요.
- 문서 근거로 에이블스쿨에서 어떤 학습 방식과 환경(이론, 실습, 프로젝트, 동료와의 협업 등)이 있는지 설명하세요.
- 지금 바로 해볼 수 있는 구체적인 행동 1~2가지를 제안하세요.
- 도전을 응원하는 따뜻한 한 문장으로 마무리하세요.

## 답변 형식
- 첫 문장에서 질문에 대한 핵심 답을 바로 제시하세요.
- 보통 3~6문장으로 답하고, 항목이 여러 개면 글머리표로 정리하세요.
- 친근한 존댓말을 쓰고, 이모지는 최대 1개만 쓰세요.
- 자연스러운 경우 마지막에 짧은 응원이나 다음 행동 제안을 덧붙이세요.

[문서]
{context}"""

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
