# app/models/ml/stall_recommender.py
import torch
import torch.nn as nn


class StallRecommender(nn.Module):
    """
    user_id, stall_category_id, numeric features 를 받아서
    추천 점수(실수 1개)를 내는 MLP 모델
    """
    def __init__(self, num_users: int, num_categories: int, numeric_dim: int = 6):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 16)
        self.cat_emb = nn.Embedding(num_categories, 8)

        self.mlp = nn.Sequential(
            nn.Linear(16 + 8 + numeric_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, cat_ids, numeric):
        """
        user_ids: LongTensor [B]
        cat_ids : LongTensor [B]
        numeric : FloatTensor [B, numeric_dim]
        return  : FloatTensor [B]
        """
        u = self.user_emb(user_ids)
        c = self.cat_emb(cat_ids)
        x = torch.cat([u, c, numeric], dim=1)
        out = self.mlp(x).squeeze(-1)
        return out