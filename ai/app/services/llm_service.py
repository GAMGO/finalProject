import os
import json
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def rank_stalls_with_llm(user_pref: str, stalls: list, top_k: int = 3):
    """
    stalls: [
      {
        "store_idx": ...,
        "name": "...",
        "distance_from_route": ...,
        "rating_avg": ...,
        "rating_count_log": ...,
        "sentiment_score": ...,
        "price_level": ...,
        "stall_category_id": ...
      },
      ...
    ]
    """
    prompt = f"""
당신은 길거리 음식/노점 추천 전문가입니다.

[사용자 취향]
{user_pref or "사용자 취향 정보 없음. 일반적인 한국 직장인 기준으로 추천."}

[후보 노점 리스트]
아래는 후보 노점들의 JSON 리스트입니다.
각 노점의 필드 의미:
- distance_from_route: 사용자의 현재 위치 또는 경로에서의 거리 (m). 작을수록 가까움.
- rating_avg: 별점 평균 (1~5).
- rating_count_log: 리뷰 수의 log 스케일 (값이 클수록 리뷰가 많고 신뢰도 높음).
- sentiment_score: 최근 리뷰의 감성 점수 (-1~1, 높을수록 긍정적).
- price_level: 가격대 (1=저렴, 3=비쌈).
- stall_category_id: 음식 카테고리 ID.

후보 노점 리스트(JSON):
{json.dumps(stalls, ensure_ascii=False)}

위 후보들 중에서 사용자가 좋아할 만한 노점 {top_k}개를 골라주세요.

출력 형식(반드시 아래 JSON 형식으로만 출력):
[
  {{
    "store_idx": <가게 ID>,
    "reason": "<이 노점을 추천하는 한글 이유>"
  }},
  ...
]
"""

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    content = resp["choices"][0]["message"]["content"]

    # JSON 파싱 시도
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except Exception:
        # 파싱 실패 시 통채로 텍스트 반환
        return content

    return content