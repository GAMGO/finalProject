# dashboard.py
import requests
import streamlit as st

API_URL = "http://localhost:8000/analyze"

st.title("⭐ 리뷰 분석 대시보드")
st.write("별점 예측 + 감성 분석 + 독성 탐지 + 요약")

text = st.text_area("리뷰 내용을 입력하세요", height=150)

if st.button("분석하기"):
    if text.strip():
        with st.spinner("분석 중..."):
            res = requests.post(API_URL, json={"text": text})
            if res.status_code == 200:
                data = res.json()
                st.subheader("예측 결과")
                st.metric("예상 별점", f"{data['rating']} 점")
                st.write(f"**감성:** {data['sentiment']}")
                st.write(f"**독성 여부:** {data['toxicity']}")
                st.subheader("요약")
                st.write(data["summary"])
            else:
                st.error(f"API 오류: {res.status_code}")
    else:
        st.warning("텍스트를 입력해주세요.")
        