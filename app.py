
import os
import re
import html
import uuid
import tempfile
from operator import itemgetter

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# 0. 기본 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="AIVLE 메이트 | 에이블스쿨 학습 도우미", page_icon="🚀", layout="wide")

# Streamlit Cloud Secrets → 환경 변수
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

CHROMA_DIR = "chroma_db"            # Colab에서 만든 벡터 저장소
COLLECTION = "aivle_docs"
DEFAULT_PDF = "AIVLE School 백서.pdf"
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png"
HISTORY_TURNS = 6                   # 대화 맥락으로 넘길 최근 메시지 수

# ─────────────────────────────────────────────
# 1. 시스템 메시지 (역할 지정)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 'AIVLE 메이트'입니다.
KT 에이블스쿨을 먼저 경험한 든든한 선배이자 학습 멘토로서, 지원을 고민하는 예비 교육생과 현재 교육생의 질문에 답합니다.

[답변 원칙]
1. 문서 기반: 사실 정보는 반드시 아래 [백서 문서]와 [FAQ 핵심 기준]에서만 가져옵니다.
   - 숫자, 기간, 일정, 조건, 명칭은 문서에 적힌 그대로 정확히 전달합니다.
   - 문서에 없는 내용은 지어내지 말고 "백서에는 이 내용이 없어요"라고 솔직히 말한 뒤, 문서에서 가장 관련 있는 정보를 안내합니다.
   - 취업 보장처럼 문서에 없는 약속은 하지 않습니다.
   - 모집 일정·자격·혜택처럼 기수마다 바뀌는 정보는 "몇 기 기준"인지 함께 밝히고, 최종 확인은 공식 홈페이지(aivle.kt.co.kr)에서 하도록 한 줄 안내합니다.
   - 선배 사례(전공, 기수, 성과)가 문서에 있으면 1~2개를 구체적으로 들어 설득력을 높입니다.
   - 문서 안의 [1], [12] 같은 출처 번호 표기는 답변에 옮기지 않습니다.

2. 답변 구조 (마크다운):
   - 첫 줄: 질문에 대한 **핵심 답변 한 문장**을 굵게 씁니다.
   - 본문: 질문 성격에 맞게 한눈에 보이도록 시각화합니다.
     · 비교 (트랙, 과정 종류, 장단점, 전/후 변화) → 마크다운 표
     · 과정, 일정, 절차, 성장 단계 → 단계 흐름 (예: 1단계 ➜ 2단계 ➜ 3단계) 과 기간이 있으면 표
     · 학습 팁, 준비 방법, 전략 → 이모지 불릿 3~5개
     · 소제목이 필요하면 #### 수준의 짧은 제목을 사용합니다.
   - 마지막 줄: "💪 " 로 시작하는 응원 한마디.
     질문자의 상황(비전공자, 직장인, 불안함, 진로 고민 등)을 짚어 구체적으로, 지금 바로 도전하고 싶어지게 씁니다.

3. 톤: 친근한 존댓말. 선배가 후배에게 말하듯 따뜻하지만 자신감 있게. 불필요한 서론 없이 바로 핵심부터.

4. 분량: 핵심 위주로 간결하게. 표나 단계 흐름은 적극 활용하되 같은 내용을 반복하지 않습니다.

5. 에이블스쿨과 관계없는 질문에는 짧게 답하고, 에이블스쿨 관련 질문으로 자연스럽게 이어 줍니다.

6. FAQ·규정 질문: 지원 자격, 교육비, 선발 절차, 모집 일정, 출결, 병행, 수료, 문의처, 취업 지원처럼 규정과 절차를 묻는 질문은 2번 답변 구조보다 아래를 우선합니다.
   - 근거: [백서 문서]와 [FAQ 핵심 기준]을 함께 씁니다. 둘이 다르거나 [백서 문서]에 일부만 보이면 [FAQ 핵심 기준]을 우선합니다.
   - 고정 표현: [FAQ 핵심 기준]에 있는 주제는 그 순서, 단계 이름, 수치를 그대로 쓰고, 단계를 더하거나 빼거나 다른 이름으로 바꾸지 않습니다.
   - 최신 기준 우선: 문서에 "다만", "바뀌어", "~이후 시작하는", "~기부터" 같은 변경 내용이 있으면 바뀐 기준을 결론으로 쓰고, 예전 기준은 "예전(8기까지)에는 ~"처럼 구분합니다.
   - 첫 줄: "네", "아니요", "조건에 따라 달라요" 중 하나로 시작하는 결론 한 문장을 굵게 씁니다. 절차를 묻는 질문은 절차 전체를 한 문장으로 요약합니다.
   - 본문: 기준(수치, 날짜, 조건)을 문서 표현 그대로 쓰고, 금액·기간 구간이 있으면 표로 정리합니다. 선배 사례는 경험을 묻는 질문에만 넣습니다.
   - 개인 상황 질문 (예: 92년생이에요, 알바 중이에요): 기준을 대입해 설명하되 "지원 가능합니다"라고 확정하지 않고, 기준과 확인처를 함께 씁니다.
   - 하지 말 것:
     · 발표되지 않은 모집 일정 예측
     · 개인별 금액을 계산해 확정하기
     · 출결·중도 포기처럼 불이익이 있는 규정을 "걱정 마세요"처럼 가볍게 넘기기
   - 확인처: 응원 줄 바로 위에 한 줄로 안내합니다. (공식 홈페이지 aivle.kt.co.kr, 고용24, 카카오채널 @aivleschool, 이메일 kt_aivle@kbrainc.com, 운영 사무국)
   - 응원 한마디: 규정 질문에서는 준비를 격려하는 방향으로 짧게 씁니다. (예: 💪 미리 확인해 두면 입교 후가 훨씬 편해져요!)

[FAQ 핵심 기준]
1. 교육비와 자기부담금: 교육비, 무료, 무상, 비용, 수강료, 자부담을 묻는 질문에 적용합니다.
   - 결론: "교육비는 대부분 정부가 지원하지만, 기수에 따라 자기부담금이 있을 수 있어요." "전액 무료", "전액 무상"이라고 단정하지 않습니다.
   - 에이블스쿨은 고용노동부 K-디지털 트레이닝(KDT) 과정이며, 교육비는 국민내일배움카드로 신청해 정부 지원을 받습니다.
   - KT의 "교육은 전액 무상" 안내와 9기 보도의 "전액 지원"은 예전 안내입니다.
   - 2026년 1월 1일 이후 시작하는 KDT 과정에는 훈련비의 10% 수준 자기부담금 규정이 신설되었습니다. 10기 교육 시작일(2026년 9월 29일)은 이 날짜 이후입니다.
   - 정부가 훈련비의 90% 이상을 지원하고, 본인 부담은 아래 구간에 따라 정해집니다.
     · 300만원 이하: 훈련비의 10%
     · 300~600만원: 30만원
     · 600~1,200만원: 40만원
     · 1,200~1,800만원: 50만원
     · 1,800만원 초과: 60만원(상한)
   - 국민취업지원제도 Ⅰ유형 참여자 등은 면제됩니다.
   - 자부담금 결제에도 국민내일배움카드를 쓰므로, 고용24에서 남은 지원 한도를 확인하도록 안내합니다.
   - 실제 금액은 고용24의 해당 과정 상세 페이지와 입교 안내에서 확인하도록 안내합니다.

2. 선발 절차와 준비: 선발 과정, 전형, 시험, 면접, 역량검사, 역량심사, 코딩테스트, 인적성을 묻는 질문에 적용합니다.
   - 결론: 선발 절차는 "지원서 접수 → 서류 발표 → 역량심사 → 합격자 발표 → 입교"의 5단계입니다. 이 순서와 단계 이름을 그대로 씁니다.
   - 10기 일정: 지원서 접수(7.7~8.10) → 서류 발표(8.14) → 역량심사(8.14~8.23) → 합격자 발표(8.28) → 입교(9.29). 일정은 기수마다 바뀝니다.
   - 역량심사 방식: 9기와 10기부터 AI 개발자·DX 컨설턴트 트랙 모두 온라인 AI 역량검사(인적성)만 봅니다. 따라서 AI 개발자 트랙 지원자도 코딩 테스트 대신 인적성 검사를 준비하면 됩니다. 8기까지는 DX 컨설턴트 트랙이 인적성 검사, AI 개발자 트랙이 코딩 테스트(인성검사 포함)를 봤습니다.
   - 면접: 2026년 11월 기준 공식 FAQ는 별도 면접 과정이 없다고 안내합니다. 선발 절차에 면접 단계를 넣거나 면접 준비를 안내하지 않습니다. 백서의 "모의면접", "면접 특강"은 수료 후 취업 지원 내용이므로 선발 절차와 섞지 않습니다.
   - 역량검사 준비 (9기 후기 기준):
     · 서류 합격 후 이메일로 역량검사 안내가 오고, 실제와 같은 환경에서 연습 기회가 5번 주어지며, 응시 기간은 약 9일이었습니다.
     · KT그룹 인적성 검사와 비슷한 온라인 환경이며, 메모와 계산도 화면 안에서 타자로만 할 수 있었습니다.
     · 도형 문제를 어려워한 사람이 많으니 기출 유형을 PC 화면으로 미리 풀어 보도록 안내합니다.
   - 지원서 준비: 선배들은 홈페이지의 인재상과 블로그 후기를 조사해 자기소개서에 녹였습니다. 왜 AI를 배우려는지, 수료 후 어떤 일을 하고 싶은지 한 문장으로 정리해 두도록 안내합니다.
   - 답변 순서: 결론 한 문장 → 5단계와 10기 일정을 표로 → 역량심사 방식 (8기까지와 9·10기 구분) → 준비 팁 → 확인처 → 응원.

[백서 문서]
{context}"""

FOLLOWUP_PROMPT = """다음은 에이블스쿨 학습 도우미와 사용자의 대화입니다.
사용자가 이어서 궁금해할 만한 후속 질문 3개를 만들어 주세요.
- 에이블스쿨 백서로 답할 수 있는 내용이어야 합니다.
- 각 질문은 30자 이내, 사용자가 직접 묻는 말투로 씁니다.
- 번호나 기호 없이 한 줄에 하나씩만 출력합니다.

[질문]
{q}

[답변]
{a}"""


# ─────────────────────────────────────────────
# 2. RAG 파이프라인
# ─────────────────────────────────────────────
def clean_text(text):
    """PDF 변환 과정에서 섞인 마크다운·HTML·출처 번호·URL 제거"""
    t = re.sub(r"<sup>.*?</sup>", "", text, flags=re.S)          # <sup>[1]</sup>
    t = re.sub(r"<[^>]+>", "", t)                                  # 기타 HTML 태그
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", t)      # [텍스트](링크) → 텍스트
    t = re.sub(r"https?://\S+", "", t)                             # 남은 URL
    t = re.sub(r"\[\d+\](\[\d+\])*", "", t)                        # [1][32] 출처 번호
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)                # # 제목 기호
    t = re.sub(r"(\*\*|__|~~|`)", "", t)                           # 굵게·코드 기호
    t = re.sub(r"^\s*\|?[\s:\-|]{3,}\|?\s*$", "", t, flags=re.M)   # 표 구분선 |---|---|
    t = t.replace("|", " · ")
    t = re.sub(r"(\s*·\s*){2,}", " · ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" +([,.)])", r"\1", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip(" ·\n")


def is_reference_chunk(text):
    """백서 끝의 「참고 자료 및 출처」처럼 URL 목록뿐인 청크는 제외"""
    return text.count("http") >= 2 or "참고 자료 및 출처" in text


def filter_docs(docs):
    seen, out = set(), []
    for d in docs:
        if is_reference_chunk(d.page_content):
            continue
        key = clean_text(d.page_content)[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def format_docs(docs):
    return "\n\n".join(clean_text(doc.page_content) for doc in docs)


def get_embedding():
    return OpenAIEmbeddings(model="text-embedding-3-small")


CONDENSE_PROMPT = """이전 대화를 참고해, 마지막 질문을 앞의 맥락 없이도 이해되는 하나의 검색용 질문으로 바꿔 주세요.
질문만 한 줄로 출력합니다.

[이전 대화]
{history}

[마지막 질문]
{input}"""


def build_chain(retriever):
    """검색 결과(docs)와 답변(answer)을 함께 돌려주는 RAG 체인"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model=MODEL, temperature=0.3, streaming=True)
    condense_llm = ChatOpenAI(model=MODEL, temperature=0)

    # "그럼 DX 트랙은요?" 같은 이어지는 질문도 백서에서 잘 찾도록 검색어 재작성
    def search_query(x):
        if not x.get("history"):
            return x["input"]
        hist = "\n".join(
            f"{'사용자' if isinstance(m, HumanMessage) else '도우미'}: {m.content[:300]}"
            for m in x["history"][-4:])
        try:
            return condense_llm.invoke(CONDENSE_PROMPT.format(history=hist, input=x["input"])).content.strip()
        except Exception:
            return x["input"]

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | prompt
        | llm
        | StrOutputParser()
    )
    return (
        RunnablePassthrough.assign(docs=RunnableLambda(search_query) | retriever | RunnableLambda(filter_docs))
        .assign(answer=answer_chain)
    )


def rag_pipeline(path):
    """01번 노트북의 rag_pipeline(): PDF 경로 → 로드 → 분할 → 임베딩 → 벡터 저장소 → RAG 체인"""
    docs = PyMuPDF4LLMLoader(path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""])
    chunks = [c for c in splitter.split_documents(docs) if not is_reference_chunk(c.page_content)]

    # 업로드 문서는 메모리 저장소에 (이름 충돌 방지용 고유 컬렉션)
    vectorstore = Chroma.from_documents(
        chunks,
        get_embedding(),
        collection_name=f"upload_{uuid.uuid4().hex[:8]}")

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})
    return build_chain(retriever)


@st.cache_resource(show_spinner="백서를 불러오는 중입니다...")
def get_default_chain():
    """앱 시작 시 기본 백서 체인 (한 번만 생성되어 모든 세션이 공유)"""
    if os.path.isdir(CHROMA_DIR):
        vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=get_embedding(),
            persist_directory=CHROMA_DIR)
        if vectorstore._collection.count() > 0:
            retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})
            return build_chain(retriever)
    if os.path.exists(DEFAULT_PDF):
        return rag_pipeline(DEFAULT_PDF)
    return None


def suggest_followups(question, answer):
    try:
        llm = ChatOpenAI(model=MODEL, temperature=0.7)
        text = llm.invoke(FOLLOWUP_PROMPT.format(q=question, a=answer[:1500])).content
        lines = [re.sub(r"^[\s\-\d\.\)•·*]+", "", l).strip().strip('"') for l in text.splitlines()]
        return [l for l in lines if l][:3]
    except Exception:
        return []


# ─────────────────────────────────────────────
# 3. 세션 상태 (벡터화가 반복되지 않도록 State에 저장)
# ─────────────────────────────────────────────
defaults = {
    "messages": [],          # {"role", "content", "sources", "followups"}
    "chain": None,
    "document_name": None,
    "pending": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.chain is None:
    try:
        st.session_state.chain = get_default_chain()
        if st.session_state.chain is not None:
            st.session_state.document_name = DEFAULT_PDF
    except Exception as e:
        st.error(f"기본 백서를 불러오지 못했습니다: {e}")


def ask(q):
    st.session_state.pending = q


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending = None


# ─────────────────────────────────────────────
# 4. 스타일
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── 다크 테마 팔레트 ── */
:root {
  --bg: #0D171C; --bg2: #111F25; --surface: #16272E; --surface2: #1C323A;
  --line: #27434C; --text: #E4EFEE; --muted: #93ABA8;
  --aivle: #1CC7B6; --aivle-dark: #13A596; --aivle-soft: rgba(28,199,182,.12);
}
.stApp { background: radial-gradient(1200px 600px at 85% -10%, #12333A 0%, var(--bg) 55%) fixed; color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1050px; }
.stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color: var(--text); }
.stApp [data-testid="stCaptionContainer"], .stApp small { color: var(--muted) !important; }
.stApp a { color: var(--aivle); }
hr { border-color: var(--line) !important; }

/* 헤더 */
.hero {
  background: linear-gradient(120deg, #0A5E58 0%, #0F8F84 50%, #17B3A4 100%);
  border: 1px solid rgba(28,199,182,.35);
  border-radius: 20px; padding: 26px 30px; margin-bottom: 18px;
  box-shadow: 0 12px 34px rgba(0,0,0,.35); position: relative; overflow: hidden;
}
.hero:after { content: "🚀"; position: absolute; right: 28px; top: 14px; font-size: 64px; opacity: .22; }
.hero h1 { color: #fff !important; font-size: 1.95rem; margin: 0 0 4px 0; padding: 0; }
.hero p { margin: 0; color: #E6FFFB !important; opacity: .92; font-size: 1.02rem; }
.badge { display: inline-block; background: rgba(0,0,0,.22); color: #DFFCF8; border-radius: 999px;
  padding: 3px 12px; font-size: .78rem; margin-bottom: 10px; }

/* 카드 */
.feature, .stat { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; }
.feature { padding: 16px 18px; height: 100%; }
.feature .ico { font-size: 1.6rem; }
.feature b { display: block; margin: 6px 0 2px; color: var(--text); }
.feature span { font-size: .86rem; color: var(--muted) !important; }
.stat { border-left: 5px solid var(--aivle); padding: 12px 14px; }
.stat .num { font-size: 1.45rem; font-weight: 800; color: var(--aivle); }
.stat .lbl { font-size: .8rem; color: var(--muted); }
.section-title { font-weight: 700; font-size: 1.08rem; margin: 22px 0 8px; color: var(--text); }

/* 버튼 */
div.stButton > button {
  border-radius: 12px; border: 1px solid var(--line); background: var(--surface); color: var(--text);
  text-align: left; transition: all .15s ease; width: 100%;
}
div.stButton > button p { color: var(--text); }
div.stButton > button:hover { border-color: var(--aivle); background: var(--surface2); transform: translateY(-1px); }
div.stButton > button:hover p { color: var(--aivle); }
div.stButton > button[kind="primary"] { background: #3A1F24; border-color: #6B3139; }
div.stButton > button[kind="primary"] p { color: #FFB4B4; }
div.stButton > button[kind="primary"]:hover { background: #4A252B; border-color: #FF8A8A; }

/* 사이드바 */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0F2228 0%, #0B1418 100%);
  border-right: 1px solid var(--line); }
.logo-card { background: rgba(255,255,255,.92); border-radius: 14px; padding: 12px 16px; margin-bottom: 8px; }
.logo-card img { width: 100%; display: block; }
.doc-chip { background: var(--aivle-soft); border: 1px solid rgba(28,199,182,.35); border-radius: 12px;
  padding: 8px 12px; font-size: .85rem; color: var(--text); }
.doc-chip b { color: var(--aivle); }

/* 입력·업로드·펼침 */
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
[data-testid="stExpander"] summary:hover p { color: var(--aivle); }
[data-testid="stFileUploaderDropzone"] { background: var(--bg2); border: 1px dashed var(--line); }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background: transparent; }
[data-testid="stChatInput"] { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; }
[data-testid="stChatInput"] textarea { color: var(--text); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted); }

/* 채팅 */
[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--line);
  border-radius: 16px; padding: 12px 16px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) { background: var(--surface2); }
[data-testid="stChatMessage"] table { width: 100%; border-collapse: collapse; }
[data-testid="stChatMessage"] th { background: rgba(28,199,182,.18); color: var(--text); }
[data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td { border: 1px solid var(--line); padding: 6px 10px; }
[data-testid="stChatMessage"] strong { color: #7FF0E3; }
[data-testid="stChatMessage"] code { background: var(--bg2); color: #9FF5EA; }
.followup-label { font-size: .85rem; color: var(--muted); margin: 6px 0 2px; }

/* 답변 근거 카드 */
.src-card { background: var(--bg2); border: 1px solid var(--line); border-left: 4px solid var(--aivle);
  border-radius: 10px; padding: 10px 14px; margin: 0 0 10px 0; }
.src-head { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.src-no { background: var(--aivle); color: #062B27 !important; font-weight: 800; font-size: .75rem;
  border-radius: 999px; width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; }
.src-page { font-size: .75rem; color: var(--aivle) !important; border: 1px solid rgba(28,199,182,.4);
  border-radius: 999px; padding: 1px 8px; }
.src-title { font-weight: 700; font-size: .9rem; color: var(--text); margin: 2px 0 4px; }
.src-text { font-size: .84rem; line-height: 1.55; color: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 5. 사이드바: 로고 / 관리자(문서 업로드) / 추천 질문 / 대화 삭제
# ─────────────────────────────────────────────
SIDEBAR_QUESTIONS = [
    "🎓 에이블스쿨은 어떤 곳인지 숫자로 한눈에 보여 주세요",
    "📊 AI 개발자 트랙과 DX 컨설턴트 트랙을 표로 비교해 주세요",
    "📝 지원 자격과 선발 절차를 단계별로 알려 주세요",
    "⏰ 하루 일정과 출석 규정은 어떻게 되나요?",
    "💼 수료 후에는 어떤 취업 지원을 받을 수 있나요?",
]

with st.sidebar:
    st.markdown(f'<div class="logo-card"><img src="{LOGO_URL}" alt="AIVLE"></div>', unsafe_allow_html=True)
    st.caption("에이블스쿨은 AI/DX 인재를 양성하는 KT의 교육 프로그램입니다.")

    st.markdown("#### 🔐 관리자")
    with st.expander("📂 백서 문서 업로드", expanded=st.session_state.chain is None):
        uploaded_file = st.file_uploader("PDF 업로드", type=["pdf"], label_visibility="collapsed")
        if uploaded_file:
            if st.session_state.document_name != uploaded_file.name:
                with st.spinner("문서를 처리하고 있습니다..."):
                    try:
                        # ① 임시 파일 생성 ② 내용 저장 ③ 경로 확보 ④ RAG 파이프라인에 '경로' 전달
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        st.session_state.chain = rag_pipeline(tmp_path)
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.messages = []
                        st.toast("새 문서로 학습 도우미가 준비되었어요! ✨")
                    except Exception as e:
                        st.error(f"문서 처리 중 오류: {e}")

    if st.session_state.document_name:
        st.markdown(f'<div class="doc-chip">📘 사용 중인 문서<br><b>{st.session_state.document_name}</b></div>',
                    unsafe_allow_html=True)
    else:
        st.warning("문서를 업로드해 주세요.")

    st.divider()
    st.markdown("#### 💡 추천 질문")
    for i, q in enumerate(SIDEBAR_QUESTIONS):
        st.button(q, key=f"side_q_{i}", on_click=ask, args=(q.split(" ", 1)[1],))

    st.divider()
    n_q = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.caption(f"💬 지금까지 나눈 질문: **{n_q}개**")
    st.button("🗑️ 대화 삭제", key="clear", on_click=clear_chat, type="primary")


# ─────────────────────────────────────────────
# 6. 헤더
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">KT AIVLE School · 백서 기반 RAG 학습 도우미</div>
  <h1>AIVLE 메이트</h1>
  <p>먼저 걸어간 선배처럼, 백서에 근거해 정확하게 답하고 여러분의 도전을 진심으로 응원합니다.</p>
</div>
""", unsafe_allow_html=True)

user_input = st.chat_input("에이블스쿨에 대해 무엇이든 물어보세요 ✨")
if st.session_state.pending and not user_input:
    user_input = st.session_state.pending
st.session_state.pending = None


# ─────────────────────────────────────────────
# 7. 첫 화면 (대화가 없을 때)
# ─────────────────────────────────────────────
WELCOME_QUESTIONS = {
    "🤔 지원을 고민하고 있다면": [
        "비전공자인데 6개월 과정을 잘 따라갈 수 있을까요?",
        "저에게 맞는 트랙은 어떻게 고르면 되나요?",
    ],
    "📚 과정이 궁금하다면": [
        "입교하면 첫 8주는 어떻게 흘러가나요?",
        "미니프로젝트와 빅프로젝트는 뭐가 다른가요?",
    ],
    "🚀 성장과 수료 후가 궁금하다면": [
        "선배들은 어떤 공부법으로 성장했나요?",
        "에이블스쿨을 수료하면 무엇을 얻을 수 있나요?",
    ],
}

# 백서 01장 「숫자로 보는 에이블스쿨」
STATS = [("840시간", "이론·실습·프로젝트 (10기)"), ("약 3,500명", "누적 수료생"),
         ("500여 개", "수료생 진출 기업"), ("4회 연속", "KDT 해커톤 대상")]

if not st.session_state.messages and not user_input:
    st.markdown('<div class="section-title">📈 숫자로 보는 에이블스쿨</div>', unsafe_allow_html=True)
    for col, (num, label) in zip(st.columns(4), STATS):
        col.markdown(f'<div class="stat"><div class="num">{num}</div><div class="lbl">{label}</div></div>',
                     unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    for col, (ico, title, desc) in zip(
        (c1, c2, c3),
        [("📘", "백서 기반 정확한 답변", "근거가 된 백서 내용을 함께 보여 드려요"),
         ("📊", "표·단계로 한눈에", "비교와 일정은 표와 흐름도로 정리해요"),
         ("💪", "도전을 응원해요", "내 상황에 맞춘 선배의 한마디까지")]):
        col.markdown(f'<div class="feature"><div class="ico">{ico}</div><b>{title}</b><span>{desc}</span></div>',
                     unsafe_allow_html=True)

    st.markdown('<div class="section-title">이런 것부터 물어보세요 👇</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (cat, qs) in zip(cols, WELCOME_QUESTIONS.items()):
        with col:
            st.markdown(f"**{cat}**")
            for j, q in enumerate(qs):
                st.button(q, key=f"welcome_{cat}_{j}", on_click=ask, args=(q,))


# ─────────────────────────────────────────────
# 8. 대화 출력
# ─────────────────────────────────────────────
AVATAR = {"user": "🧑‍🎓", "assistant": "🤖"}


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 답변 근거 · 백서 발췌 {len(sources)}건"):
        cards = []
        for i, s in enumerate(sources, 1):
            page = f'<span class="src-page">p.{s["page"]}</span>' if s.get("page") else ""
            title = f'<div class="src-title">{html.escape(s["title"])}</div>' if s.get("title") else ""
            cards.append(
                f'<div class="src-card"><div class="src-head"><span class="src-no">{i}</span>{page}</div>'
                f'{title}<div class="src-text">{html.escape(s["text"])}</div></div>')
        st.markdown("".join(cards), unsafe_allow_html=True)


def to_sources(docs):
    out = []
    for d in docs or []:
        page = d.metadata.get("page")
        if isinstance(page, int):
            page += 1
        raw = d.page_content
        # 청크 안의 첫 번째 제목을 소제목으로 사용
        m = re.search(r"^\s*#{1,6}\s*(.+)$", raw, flags=re.M)
        title = clean_text(m.group(1))[:60] if m else ""
        text = clean_text(raw).replace("\n", " ")
        if title and text.startswith(title):
            text = text[len(title):].strip()
        if not text:
            continue
        out.append({"page": page, "title": title,
                    "text": text[:260] + ("…" if len(text) > 260 else "")})
    return out[:4]


last_idx = len(st.session_state.messages) - 1
for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"], avatar=AVATAR[m["role"]]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources"))
    # 마지막 답변 아래에 이어서 물어볼 질문
    if idx == last_idx and m["role"] == "assistant" and m.get("followups") and not user_input:
        st.markdown('<div class="followup-label">👉 이어서 물어보기</div>', unsafe_allow_html=True)
        fcols = st.columns(len(m["followups"]))
        for j, (fc, fq) in enumerate(zip(fcols, m["followups"])):
            fc.button(fq, key=f"follow_{idx}_{j}", on_click=ask, args=(fq,))


# ─────────────────────────────────────────────
# 9. 질문 처리
# ─────────────────────────────────────────────
if user_input:
    with st.chat_message("user", avatar=AVATAR["user"]):
        st.markdown(user_input)

    if st.session_state.chain is None:
        with st.chat_message("assistant", avatar=AVATAR["assistant"]):
            st.warning("먼저 왼쪽 '관리자' 메뉴에서 백서 PDF를 업로드해 주세요.")
        st.stop()

    history = []
    for m in st.session_state.messages[-HISTORY_TURNS:]:
        history.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))

    holder = {}

    def stream_answer():
        for chunk in st.session_state.chain.stream({"input": user_input, "history": history}):
            if "docs" in chunk:
                holder["docs"] = chunk["docs"]
            if "answer" in chunk:
                yield chunk["answer"]

    with st.chat_message("assistant", avatar=AVATAR["assistant"]):
        try:
            with st.spinner("백서에서 관련 내용을 찾고 있어요... 🔎"):
                answer = st.write_stream(stream_answer())
        except Exception as e:
            answer = f"⚠️ 답변 생성 중 문제가 발생했어요: {e}"
            st.error(answer)
        sources = to_sources(holder.get("docs"))
        render_sources(sources)

    with st.spinner("이어서 물어볼 질문을 고르는 중..."):
        followups = suggest_followups(user_input, answer if isinstance(answer, str) else "")

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant", "content": answer if isinstance(answer, str) else str(answer),
        "sources": sources, "followups": followups,
    })
    st.rerun()
