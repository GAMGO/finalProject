import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def safe_float(v):
    try:
        return float(v)
    except:
        return None


def generate_recommend_reason(store: dict, origin: str) -> str:
    name = store.get("store_name") or store.get("name") or "이 가게"

    # distance_m을 안전하게 float 변환
    distance_val = safe_float(store.get("distance_m"))
    distance_str = f"{distance_val:.2f}" if distance_val is not None else "정보없음"

    prompt = f"""
다음 가게를 사용자의 경로({origin}) 기준으로 추천하려고 합니다.
가게 정보:
- 이름: {name}
- 별점: {store.get('rating', '정보없음')}
- 감성 점수(sentiment_score): {store.get('sentiment_score', '정보없음')}
- 거리(m): {distance_str}

위 정보를 바탕으로 1문장의 자연스러운 추천 사유를 작성해줘.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 맛집 추천 전문가입니다."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=80,
    )

    return completion.choices[0].message.content.strip()