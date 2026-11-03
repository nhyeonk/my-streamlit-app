%%writefile app.py

import os
import streamlit as st
from openai import OpenAI

# 페이지 설정
st.set_page_config(
    page_title="에이블 AI 어시스턴트",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ 에이블 AI 어시스턴트")
st.caption("텍스트 및 음성으로 회의 내용을 질문하는 AI 비서")

# API 키 설정
# Streamlit Cloud에서는 Secrets에 OPENAI_API_KEY 등록
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

if not api_key:
    st.error("OpenAI API 키를 설정해 주세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# 회의 내용 불러오기
@st.cache_data
def load_meeting():
    if os.path.exists("meeting.txt"):
        with open("meeting.txt", "r", encoding="utf-8") as f:
            return f.read()
    return ""

meeting_text = load_meeting()

# 시스템 역할 설정
sys_role = """
당신은 회의 내용을 분석하는 AI 비서입니다.

규칙:
1. 사용자의 질문에 친절하고 정확하게 답변합니다.
2. 회의 내용과 관련된 질문은 제공된 회의록을 근거로 답변합니다.
3. 회의록에 없는 내용은 추측하지 않습니다.
4. 중요한 결정 사항과 담당자, 일정 등을 명확하게 정리합니다.
5. 한국어로 답변합니다.
"""

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "안녕하세요! 회의 내용에 대해 질문해 주세요."
        }
    ]

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")

    if meeting_text:
        st.success("회의록을 불러왔습니다.")
    else:
        st.warning("meeting.txt 파일이 없습니다.")

    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "대화가 초기화되었습니다."
            }
        ]
        st.rerun()

# 기존 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 음성 입력
st.subheader("🎤 음성 질문")

audio_file = st.audio_input(
    "마이크를 눌러 질문을 녹음하세요.",
    sample_rate=16000
)

voice_question = None

if audio_file is not None:

    # 같은 녹음이 재실행 시 중복 처리되지 않도록 확인
    import hashlib
    audio_bytes = audio_file.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    if st.session_state.get("last_audio_hash") != audio_hash:
        with st.spinner("음성을 텍스트로 변환 중입니다..."):
            try:
                audio_file.seek(0)

                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=audio_file,
                    language="ko"
                )

                voice_question = transcript.text.strip()
                st.session_state.last_audio_hash = audio_hash

                if voice_question:
                    st.success(f"🎤 인식된 질문: {voice_question}")
                else:
                    st.warning("인식된 음성이 없습니다.")

            except Exception as e:
                st.error(f"음성 인식 오류: {e}")

# 텍스트 입력
user_input = st.chat_input("메시지를 입력하세요...")

# 텍스트 또는 음성 입력 선택
question = voice_question or user_input

if question:

    # 사용자 질문 저장
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # GPT 응답 생성
    with st.chat_message("assistant"):

        with st.spinner("에이블이 분석하고 있습니다..."):
            try:
                # 회의록은 시스템 지침과 별도로 전달
                context = (
                    f"다음은 참고할 회의록입니다.\n\n{meeting_text}"
                    if meeting_text else
                    "현재 제공된 회의록이 없습니다."
                )

                response = client.responses.create(
                    model="gpt-4.1-mini",
                    temperature=0,
                    instructions=sys_role + "\n\n" + context,
                    input=[
                        m for m in st.session_state.messages
                        if m["role"] in ("user", "assistant")
                    ]
                )

                assistant_reply = response.output_text

            except Exception as e:
                st.error(f"응답 생성 오류: {e}")
                st.stop()

        st.markdown(assistant_reply)

    # 답변 저장
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
