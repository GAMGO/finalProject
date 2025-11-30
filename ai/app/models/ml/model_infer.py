# app/models/ml/model_infer.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def predict_scores(model, df, user2idx, cat2idx, device="cpu"):
    user_ids = torch.tensor([user2idx[u] for u in df["user_id"]], dtype=torch.long, device=device)
    cat_ids = torch.tensor([cat2idx[c] for c in df["stall_category_id"]], dtype=torch.long, device=device)

    numeric_cols = [
        "distance_from_route",
        "rating_avg",
        "rating_count_log",
        "sentiment_score",
        "price_level",
        "hour_sin",
    ]

    numeric = torch.tensor(df[numeric_cols].values, dtype=torch.float32, device=device)
    
    scores = model(user_ids, cat_ids, numeric)
    return scores.cpu().numpy()