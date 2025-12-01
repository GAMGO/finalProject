# ai/app/models/model_infer.py
import torch
from app.models.stall_recommender import StallRecommender

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_recommender_model(path: str = "stall_recommender.pt") -> StallRecommender:
    state = torch.load(path, map_location=device)

    num_users = state["user_emb.weight"].shape[0]
    num_stores = state["store_emb.weight"].shape[0]

    print(f"[INFO] Loading model with num_users={num_users}, num_stores={num_stores}")

    model = StallRecommender(num_users=num_users, num_stores=num_stores)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# 글로벌 모델 인스턴스 (FastAPI import 시 한 번만 로드)
model = load_recommender_model()


def _to_tensor(v, dtype):
    return torch.tensor([v], dtype=dtype, device=device)


def predict_store_score(user_id: int, store: dict) -> float:
    """
    store 예시:
    {
      "idx": 123,                 # store_idx
      "sentiment_score": 0.8,     # 이 가게 리뷰들의 평균 감성 점수 등
      "rating": 4.5,              # 평균 별점
      "distance": 120.0,          # m 단위 거리 등
      "hour_sin": 0.5,            # 현재 시간 기준 sin 인코딩
    }
    """
    batch = {
        "user_id": _to_tensor(user_id, torch.long),
        "store_id": _to_tensor(store["idx"], torch.long),
        "sentiment": _to_tensor(store["sentiment_score"], torch.float32),
        "rating": _to_tensor(store["rating"], torch.float32),
        "distance": _to_tensor(store["distance"], torch.float32),
        "hour_sin": _to_tensor(store["hour_sin"], torch.float32),
    }

    with torch.no_grad():
        score = model(batch).cpu().item()

    return float(score)


def predict_scores(user_id: int, stores: list[dict]) -> list[float]:
    """
    여러 가게에 대해 한 번에 점수 예측
    stores: 위 단일 store dict의 리스트
    """
    if not stores:
        return []

    user_ids = torch.tensor(
        [user_id] * len(stores), dtype=torch.long, device=device
    )
    store_ids = torch.tensor(
        [s["idx"] for s in stores], dtype=torch.long, device=device
    )
    sentiments = torch.tensor(
        [s["sentiment_score"] for s in stores], dtype=torch.float32, device=device
    )
    ratings = torch.tensor(
        [s["rating"] for s in stores], dtype=torch.float32, device=device
    )
    distances = torch.tensor(
        [s["distance"] for s in stores], dtype=torch.float32, device=device
    )
    hour_sins = torch.tensor(
        [s["hour_sin"] for s in stores], dtype=torch.float32, device=device
    )

    batch = {
        "user_id": user_ids,
        "store_id": store_ids,
        "sentiment": sentiments,
        "rating": ratings,
        "distance": distances,
        "hour_sin": hour_sins,
    }

    with torch.no_grad():
        scores = model(batch).cpu().numpy().tolist()

    return [float(s) for s in scores]