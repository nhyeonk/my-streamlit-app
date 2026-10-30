
import os
import base64
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client =

# 페이지 설정
st.set_page_config(     )
st.title(     )

# 텍스트 입력
text = st.text_area(
    "읽을 텍스트를 입력하세요",
    "안녕하세요. OpenAI 텍스트 음성 변환 실습입니다."
)

# 버튼
if st.button(      ):
    if text.strip() == "":
        st.warning(      )
    else:
        # TTS 생성
        speech = client.audio.speech.create(
            model=     ,
            voice=     ,
            input=
        )
        audio_bytes =

        # 음성 출력
        st.markdown("### 🔊 음성 재생")
        st.audio(speech.content, format="audio/mp3")
