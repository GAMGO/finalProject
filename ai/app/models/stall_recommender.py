import torch
import torch.nn as nn

class StallRecommender(nn.Module):
    def __init__(self, num_users: int, num_stores: int):
        super().__init__()

        # 유저 / 가게 임베딩
        self.user_emb = nn.Embedding(num_users, 32)
        self.store_emb = nn.Embedding(num_stores, 32)

        # float feature 개수: sentiment_score, rating, distance, hour_sin
        float_feature_dim = 4

        self.mlp = nn.Sequential(
            nn.Linear(32 + 32 + float_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, batch: dict):
        """
        batch keys:
          - user_id: (B,) long
          - store_id: (B,) long
          - sentiment: (B,) float
          - rating: (B,) float
          - distance: (B,) float
          - hour_sin: (B,) float
        """
        user_e = self.user_emb(batch["user_id"])    # (B, 32)
        store_e = self.store_emb(batch["store_id"]) # (B, 32)

        float_features = torch.stack([
            batch["sentiment"],
            batch["rating"],
            batch["distance"],
            batch["hour_sin"],
        ], dim=1)  # (B, 4)

        x = torch.cat([user_e, store_e, float_features], dim=1)  # (B, 68)
        out = self.mlp(x)  # (B, 1)
        return out.squeeze(-1)  # (B,)
