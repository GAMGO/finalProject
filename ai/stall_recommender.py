import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random


# ============================================================
# 1) 딥러닝 모델 정의 (User Embedding + Category Embedding + MLP)
# ============================================================
class StallRankingModel(nn.Module):
    def __init__(self, num_users, num_categories, user_emb_dim=16, cat_emb_dim=8):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.cat_emb = nn.Embedding(num_categories, cat_emb_dim)

        # numeric feature 개수
        self.numeric_dim = 8  # 아래 정의된 수치 피처 8개

        input_dim = user_emb_dim + cat_emb_dim + self.numeric_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # score 출력
        )

    def forward(self, user_id, cat_id, numeric):
        u = self.user_emb(user_id)         # [B, dim]
        c = self.cat_emb(cat_id)           # [B, dim]
        x = torch.cat([u, c, numeric], dim=1)
        return self.mlp(x).squeeze(-1)     # [B]


# ============================================================
# 2) Dataset 정의
# ============================================================
class StallDataset(Dataset):
    def __init__(self, df, user2idx, cat2idx):
        self.df = df.reset_index(drop=True)
        self.user2idx = user2idx
        self.cat2idx = cat2idx

        self.numeric_cols = [
            "distance_from_route",
            "detour_time_min",
            "rating_avg",
            "rating_count_log",
            "sentiment_score",
            "price_level",
            "hour_sin",
            "hour_cos",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user_id = torch.tensor(self.user2idx[row["user_id"]], dtype=torch.long)
        cat_id = torch.tensor(self.cat2idx[row["stall_category_id"]], dtype=torch.long)

        numeric = torch.tensor(row[self.numeric_cols].values, dtype=torch.float32)

        label = torch.tensor(row["label"], dtype=torch.float32)

        return user_id, cat_id, numeric, label


# ============================================================
# 3) 학습 함수
# ============================================================
def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for user_id, cat_id, numeric, label in train_loader:
            user_id, cat_id, numeric, label = (
                user_id.to(device),
                cat_id.to(device),
                numeric.to(device),
                label.to(device),
            )

            optimizer.zero_grad()

            logits = model(user_id, cat_id, numeric)
            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ----- validation -----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user_id, cat_id, numeric, label in val_loader:
                user_id, cat_id, numeric, label = (
                    user_id.to(device),
                    cat_id.to(device),
                    numeric.to(device),
                    label.to(device),
                )
                logits = model(user_id, cat_id, numeric)
                loss = criterion(logits, label)
                val_loss += loss.item()

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model


# ============================================================
# 4) 추천(추첨) 함수 - softmax 가중치 랜덤 선별
# ============================================================
@torch.no_grad()
def recommend(model, candidates_df, user_id, user2idx, cat2idx, k=3, temperature=0.7, device="cpu"):
    model.eval()

    numeric_cols = [
        "distance_from_route",
        "detour_time_min",
        "rating_avg",
        "rating_count_log",
        "sentiment_score",
        "price_level",
        "hour_sin",
        "hour_cos",
    ]

    user_idx = torch.tensor([user2idx[user_id]] * len(candidates_df), dtype=torch.long, device=device)
    cat_idx = torch.tensor([cat2idx[x] for x in candidates_df["stall_category_id"]], dtype=torch.long, device=device)
    numeric = torch.tensor(candidates_df[numeric_cols].values, dtype=torch.float32, device=device)

    scores = model(user_idx, cat_idx, numeric)

    # softmax 확률 계산
    probs = F.softmax(scores / temperature, dim=0)

    # 가중치 샘플링
    chosen_idx = torch.multinomial(probs, k, replacement=False)

    candidates_df["score"] = scores.cpu().numpy()
    candidates_df["prob"] = probs.cpu().numpy()

    return candidates_df.iloc[chosen_idx.cpu().numpy()]


# ============================================================
# 5) 실행용 더미 데이터 생성 + 전체 파이프라인 테스트
# ============================================================
def build_dummy_data(num_users=20, num_stalls=40):
    data = []
    random.seed(0)
    np.random.seed(0)

    for _ in range(2000):  # 로그 데이터 2000개
        user = random.randint(1, num_users)
        stall = random.randint(1, num_stalls)
        cat = random.randint(1, 5)

        distance = np.random.uniform(5, 200)
        detour = np.random.uniform(0.1, 8)
        rating = np.random.uniform(2, 5)
        count = np.random.uniform(1, 300)
        senti = np.random.uniform(-1, 1)
        price = np.random.randint(1, 4)

        hour = random.randint(0, 23)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # label = 방문 여부
        label = 1 if distance < 80 and rating > 4 else 0

        data.append([
            user, stall, cat,
            distance, detour, rating, np.log1p(count),
            senti, price, hour_sin, hour_cos,
            label
        ])

    df = pd.DataFrame(data, columns=[
        "user_id", "stall_id", "stall_category_id",
        "distance_from_route", "detour_time_min",
        "rating_avg", "rating_count_log",
        "sentiment_score", "price_level",
        "hour_sin", "hour_cos",
        "label"
    ])
    return df


# ============================================================
# 6) 엔드투엔드 실행
# ============================================================
if __name__ == "__main__":
    print("=== Dummy 데이터 생성 ===")
    df = build_dummy_data()

    # ID 매핑
    user2idx = {u: i for i, u in enumerate(df["user_id"].unique())}
    cat2idx = {c: i for i, c in enumerate(df["stall_category_id"].unique())}

    # Train / Val split
    df_train = df.sample(frac=0.8, random_state=0)
    df_val = df.drop(df_train.index)

    train_ds = StallDataset(df_train, user2idx, cat2idx)
    val_ds = StallDataset(df_val, user2idx, cat2idx)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 생성
    model = StallRankingModel(
        num_users=len(user2idx),
        num_categories=len(cat2idx)
    ).to(device)

    print("=== 모델 학습 시작 ===")
    train_model(model, train_loader, val_loader, epochs=5, device=device)

    # ----------------------------------------------------------
    # 실제 추천 테스트 (경로 주변 후보 10개)
    # ----------------------------------------------------------
    print("\n=== 추천 테스트 ===")

    # 임의로 user_id = 3이 경로 잡았다고 가정
    user_id = 3

    # 후보 노점 10개 생성
    candidates = df.sample(10).copy()
    candidates["user_id"] = user_id  # 모두 같은 유저라고 가정

    recommended = recommend(
        model=model,
        candidates_df=candidates,
        user_id=user_id,
        user2idx=user2idx,
        cat2idx=cat2idx,
        k=3,
        temperature=0.7,
        device=device
    )

    print("\n=== 최종 추천된 노점 3개 ===")
    print(recommended[["stall_id", "score", "prob", "distance_from_route", "rating_avg"]])