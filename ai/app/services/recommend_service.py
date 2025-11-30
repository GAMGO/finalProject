# app/services/recommend_service.py
import torch
import torch.nn.functional as F
import pandas as pd
from app.models.ml.model_infer import predict_scores

def weighted_recommend(df, model, user_id, user2idx, cat2idx, top_k=5, temperature=0.7):
    scores = predict_scores(model, df, user2idx, cat2idx)
    df = df.copy()
    df["score"] = scores

    scores_tensor = torch.tensor(scores)
    probs = F.softmax(scores_tensor / temperature, dim=0).cpu().numpy()
    df["prob"] = probs

    chosen_idx = torch.multinomial(torch.tensor(probs), top_k, replacement=False).numpy()

    return df.iloc[chosen_idx]