# app/services/summarizer.py
from openai import OpenAI
from app.utils.chunk import chunk_reviews

client = OpenAI()


# 1) 청크 요약
def summarize_chunk(text: str) -> str:
    if not text.strip():
        return ""
    
    prompt = (
        "아래 리뷰들을 핵심만 요약해줘.\n\n"
        f"{text}"
    )

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return res.choices[0].message.content.strip()


# 2) 전체 리뷰 종합 요약
def summarize_reviews(reviews: list) -> str:
    if not reviews:
        return "아직 리뷰가 없습니다."

    chunks = chunk_reviews(reviews)

    mid_summaries = [summarize_chunk(c) for c in chunks]

    final_prompt = f"""
아래는 리뷰 청크별 요약입니다.

👉 반드시 아래 내용을 종합해 최종 1문장 요약을 만들어 주세요:
- 전체 리뷰의 평균 평점 분위기 (rating)
- 감정 점수(sentiment_score) 및 감정 라벨(sentiment_label)
- 공통적으로 언급되는 장점/단점
- 리뷰 전반의 정서적 분위기(긍정/부정/무난 등)

출력은 자연스러운 한국어 한 문장으로만 해주세요.

[청크 요약들]
{chr(10).join(mid_summaries)}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.4
    )

    return res.choices[0].message.content.strip()