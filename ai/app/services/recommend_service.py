# app/services/recommend_service.py
import torch
import torch.nn.functional as F
import pandas as pd
from app.models.model_infer import predict_scores





def weighted_recommend(df, model, user_id, user2idx, cat2idx, top_k=5, temperature=0.7):
    scores = predict_scores(model, df, user2idx, cat2idx)
    df = df.copy()
    df["score"] = scores

    scores_tensor = torch.tensor(scores)
    probs = F.softmax(scores_tensor / temperature, dim=0).cpu().numpy()
    df["prob"] = probs

    chosen_idx = torch.multinomial(torch.tensor(probs), top_k, replacement=False).numpy()

    return df.iloc[chosen_idx]

def recommend_by_user_and_candidates(user_id: int, candidate_stores: list[dict], top_k: int = 5):
    """
    candidate_stores: DB/카카오 API 등에서 모은 가게 리스트
    각 요소 예:
      {
        "idx": 123,
        "name": "...",
        "lat": ...,
        "lng": ...,
        "sentiment_score": 0.7,
        "rating": 4.2,
        "distance": 120.0,
        "hour_sin": 0.5,
      }
    """
    scores = predict_scores(user_id, candidate_stores)

    for store, score in zip(candidate_stores, scores):
        store["reco_score"] = score

    # 점수 높은 순으로 top_k 리턴
    candidate_stores.sort(key=lambda s: s["reco_score"], reverse=True)
    return candidate_stores[:top_k]


def recommend_near_point(user_id: int, stores: list, top_k: int = 5):
    """
    특정 지점 근처의 상점 리스트(stores)를 받아
    개인화 추천 점수를 계산하여 top_k 반환하는 함수.

    stores 구조 예:
    [
        {
            "idx": 123,
            "lat": 37.12,
            "lng": 127.22,
            "sentiment_score": 0.8,
            "rating": 4.2,
            "distance": 120.0,
            "hour_sin": 0.5,
        },
        ...
    ]
    """

    if not stores:
        return []

    scores = predict_scores(user_id, stores)

    # 점수 저장
    for s, sc in zip(stores, scores):
        s["reco_score"] = sc

    # 점수 높은 순으로 top_k 추천
    stores.sort(key=lambda x: x["reco_score"], reverse=True)

    return stores[:top_k]