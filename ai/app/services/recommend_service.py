# app/services/recommend_service.py
import torch
import torch.nn.functional as F
import pandas as pd
from app.models.model_infer import predict_scores
from typing import List, Dict, Any, Optional
from app.services.llm_reason_service import generate_recommend_reason
from app.utils.distance import calculate_distance
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

def recommend_near_point(user_id, stores, lat, lng, point_type, limit=5):
    if not stores:
        return []

    ml_scores = predict_scores(user_id, stores)

    for s, ml in zip(stores, ml_scores):
        s["ml_score"] = ml

        sentiment = float(s.get("sentiment_score", 0.0))
        rating = float(s.get("rating", 0.0))
        distance_m = float(s.get("distance_m", 0.0))

        s["base_score"] = (
            ml * 0.4
            + sentiment * 0.1
            + (1 / (1 + distance_m)) * 0.1
            + (rating / 5.0) * 0.3
        )

    stores_sorted = sorted(stores, key=lambda x: x["base_score"], reverse=True)
    top_items = stores_sorted[:limit]

    # 🔥 추천 기준: rating ≥ 2.0
    for s in top_items:
        rating = float(s.get("rating", 0.0))
        s["recommended"] = s["base_score"] >= 0.6

    return top_items

def strip_reason_for_unrecommended(stores):
    for s in stores:
        try:
            rating_val = float(s.get("rating", 0.0))
        except:
            rating_val = 0.0

        is_recommended = bool(s.get("recommended", True))  # 🔥 recommended 없으면 True 처리

        ok = (rating_val >= 3.0) and is_recommended

        if not ok:
            s.pop("reason", None)

    return stores