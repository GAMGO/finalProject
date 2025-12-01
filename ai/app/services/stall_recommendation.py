# app/services/stall_recommendation.py
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app.config.database import get_db_connection
from app.models.stall_recommender import StallRecommender
from app.services.llm_service import rank_stalls_with_llm

# ===== PyTorch 모델 로딩 (lazy) =====
_MODEL = None
_USER2IDX = None
_CAT2IDX = None
_NUMERIC_COLS = None

def _load_model_if_needed():
    global _MODEL, _USER2IDX, _CAT2IDX, _NUMERIC_COLS
    if _MODEL is not None:
        return

    # 현재 파일: app/services/stall_recommendation.py
    # 프로젝트 루트: .../finalProject
    root_dir = Path(__file__).resolve().parents[2]
    ckpt_path = root_dir / "ai" / "model_weights.pth"

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    user2idx = ckpt["user2idx"]
    cat2idx = ckpt["cat2idx"]
    numeric_cols = ckpt["numeric_cols"]

    model = StallRecommender(
        num_users=len(user2idx),
        num_categories=len(cat2idx),
        numeric_dim=len(numeric_cols),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _MODEL = model
    _USER2IDX = user2idx
    _CAT2IDX = cat2idx
    _NUMERIC_COLS = numeric_cols


# ===== 반경 내 노점 검색 (거리 계산 SQL) =====
def get_near_stalls(lat: float, lng: float, radius_m: int):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    idx AS store_idx,
                    name,
                    lat,
                    lng,
                    stall_category_id,
                    rating_avg,
                    rating_count_log,
                    price_level,
                    COALESCE(sentiment_score, 0) AS sentiment_score,
                    (6371000 * ACOS(
                        COS(RADIANS(%s)) * COS(RADIANS(lat)) *
                        COS(RADIANS(lng) - RADIANS(%s)) +
                        SIN(RADIANS(%s)) * SIN(RADIANS(lat))
                    )) AS distance_m
                FROM stores
                HAVING distance_m <= %s
                ORDER BY distance_m ASC
            """, (lat, lng, lat, radius_m))
            rows = cur.fetchall()
            return rows or []
    finally:
        conn.close()


# ===== LLM + PyTorch 하이브리드 추천 =====
def recommend_hybrid_for_point(
    lat: float,
    lng: float,
    radius_m: int,
    user_id: Optional[int] = None,
    user_pref: Optional[str] = None,
    top_n_model: int = 10,
    top_k_llm: int = 3,
):
    _load_model_if_needed()
    model = _MODEL
    user2idx = _USER2IDX
    cat2idx = _CAT2IDX
    numeric_cols = _NUMERIC_COLS

    rows = get_near_stalls(lat, lng, radius_m)
    if not rows:
        return []

    df = pd.DataFrame(rows)

    # numeric feature 채우기
    df["distance_from_route"] = df["distance_m"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # user embedding 선택 (없으면 첫 유저 사용)
    if user_id is not None and user_id in user2idx:
        u_idx = user2idx[user_id]
    else:
        # 아무거나 하나 (개인화 없는 fallback)
        u_idx = list(user2idx.values())[0]

    user_tensor = torch.tensor([u_idx] * len(df), dtype=torch.long)
    cat_list = df["stall_category_id"].tolist()
    cat_idx_list = [cat2idx.get(c, list(cat2idx.values())[0]) for c in cat_list]
    cat_tensor = torch.tensor(cat_idx_list, dtype=torch.long)

    numeric = torch.tensor(df[numeric_cols].values, dtype=torch.float32)

    with torch.no_grad():
        scores = model(user_tensor, cat_tensor, numeric)
    df["score"] = scores.numpy()

    # PyTorch 상위 N개 후보
    df_sorted = df.sort_values("score", ascending=False).head(top_n_model)

    candidates = df_sorted[
        ["store_idx", "name", "distance_from_route", "rating_avg",
         "rating_count_log", "sentiment_score", "price_level", "stall_category_id"]
    ].to_dict(orient="records")

    # LLM 재랭킹
    llm_result = rank_stalls_with_llm(user_pref=user_pref or "", stalls=candidates, top_k=top_k_llm)

    # llm_result가 JSON list일 경우 store_idx 매핑
    if isinstance(llm_result, list):
        # reason과 함께 merge
        reason_by_id = {item["store_idx"]: item.get("reason", "") for item in llm_result if "store_idx" in item}
        final_list = []
        for c in candidates:
            sid = c["store_idx"]
            if sid in reason_by_id:
                c["reason"] = reason_by_id[sid]
                final_list.append(c)
        return final_list

    # 파싱 실패 시: 그냥 PyTorch 상위 N개 반환 + raw LLM 응답 포함
    return {
        "candidates": candidates,
        "llm_raw": llm_result,
    }