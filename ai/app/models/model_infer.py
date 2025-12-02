# ai/app/models/model_infer.py
import math
import time
from pathlib import Path
from typing import List, Dict, Any

import torch

from app.models.stall_recommender import StallRecommender

_MODEL = None
_USER2IDX = None
_STORE2IDX = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ckpt_path() -> Path:
    """
    stall_recommender.pt의 위치를 자동으로 계산.
    현재 파일(ai/app/models/model_infer.py) 기준 부모 2단계 상위가 ai/
    """
    return Path(__file__).resolve().parents[2] / "stall_recommender.pt"


def load_recommender_model():
    """체크포인트 로드: state_dict + user2idx + store2idx"""
    global _MODEL, _USER2IDX, _STORE2IDX

    ckpt_path = _ckpt_path()
    if not ckpt_path.exists():
        raise RuntimeError(f"❌ 모델 체크포인트 없음: {ckpt_path}")

    print("[INFO] Loading recommender model...")

    ckpt = torch.load(str(ckpt_path), map_location=_DEVICE)

    if "state_dict" not in ckpt or "user2idx" not in ckpt or "store2idx" not in ckpt:
        raise RuntimeError("❌ checkpoint 포맷이 잘못됨. 새 학습 스크립트로 다시 학습해야 함.")

    state_dict = ckpt["state_dict"]
    user2idx = ckpt["user2idx"]
    store2idx = ckpt["store2idx"]

    num_users = len(user2idx)
    num_stores = len(store2idx)

    print(f"[INFO] Loaded mapping: num_users={num_users}, num_stores={num_stores}")

    model = StallRecommender(num_users=num_users, num_stores=num_stores)
    model.load_state_dict(state_dict)
    model.to(_DEVICE)
    model.eval()

    _MODEL = model
    _USER2IDX = user2idx
    _STORE2IDX = store2idx

    return model


def _ensure_model_loaded():
    """lazy-load: 처음 사용할 때만 load"""
    if _MODEL is None:
        load_recommender_model()


def predict_scores(
    user_pk: int | None,
    stores: List[Dict[str, Any]],
) -> List[float]:
    """
    여러 store에 대해 relevance score(0~1)를 계산
    store dict에 다음 key 존재해야 함:
      - idx
      - sentiment_score
      - rating
      - distance_m
    """
    _ensure_model_loaded()

    scores = []

    for s in stores:
        store_pk = int(s["idx"])
        sentiment = float(s.get("sentiment_score", 0.0))
        rating = float(s.get("rating", 0.0))
        distance_m = float(s.get("distance_m", 0.0))

        hour = time.localtime().tm_hour
        hour_sin = math.sin((hour / 24) * 2 * math.pi)

        # --- PK → idx 변환 ---
        if user_pk is not None and user_pk in _USER2IDX:
            user_idx = _USER2IDX[user_pk]
        else:
            user_idx = 0  # cold-start fallback

        if store_pk in _STORE2IDX:
            store_idx = _STORE2IDX[store_pk]
        else:
            store_idx = 0  # cold-start fallback

        batch = {
            "user_id": torch.tensor([user_idx], dtype=torch.long, device=_DEVICE),
            "store_id": torch.tensor([store_idx], dtype=torch.long, device=_DEVICE),
            "sentiment": torch.tensor([sentiment], dtype=torch.float32, device=_DEVICE),
            "rating": torch.tensor([rating], dtype=torch.float32, device=_DEVICE),
            "distance": torch.tensor([distance_m], dtype=torch.float32, device=_DEVICE),
            "hour_sin": torch.tensor([hour_sin], dtype=torch.float32, device=_DEVICE),
        }

        with torch.no_grad():
            score = _MODEL(batch).item()

        scores.append(float(score))

    return scores