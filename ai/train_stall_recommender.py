# ai/train_stall_recommender.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from app.repositories.stall_training_repository import get_stall_training_data
from datasets.stall_dataset import StallTrainDataset
from app.models.stall_recommender import StallRecommender


# --------------------------------------------------------
# store_id 컬럼 자동 탐지
# --------------------------------------------------------
def find_store_id_column(df: pd.DataFrame):
    candidates = ["store_idx", "store_id", "storeId", "idx",
                  "STORE_IDX", "STORE_ID", "STOREID"]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError("❌ DataFrame에 store id 컬럼이 없습니다.")


# --------------------------------------------------------
# label 자동 생성
# rating / sentiment_score / sentiment_label 기반
# --------------------------------------------------------
def generate_label(df: pd.DataFrame):
    if "label" in df.columns:
        print("✔ label 컬럼 이미 존재 → 그대로 사용")
        return df

    print("⚙ label 자동 생성 시작…")

    # Case 1 — rating 기반
    if "rating" in df.columns:
        print("→ rating 기반 label 생성 (rating>=4 → 1, else→0)")
        df["label"] = df["rating"].apply(lambda r: 1 if r >= 4 else 0)
        return df

    # Case 2 — sentiment_label 기반
    if "sentiment_label" in df.columns:
        print("→ sentiment_label 기반 label 생성")
        mapping = {"positive": 1, "negative": 0}
        df["label"] = df["sentiment_label"].map(mapping)
        # neutral or NaN 제거
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        return df

    # Case 3 — sentiment_score 기반
    if "sentiment_score" in df.columns:
        print("→ sentiment_score 기반 label 생성 (>=0.6 →1, <=0.4 →0)")
        df["label"] = df["sentiment_score"].apply(
            lambda s: 1 if (s is not None and s >= 0.6) else 0
        )
        return df

    raise KeyError("❌ label 생성 실패: rating / sentiment_label / sentiment_score 중 어떤 것도 없습니다.")


# --------------------------------------------------------
# 메인 학습 함수
# --------------------------------------------------------
def main():
    print("📥 학습 데이터 로드 중…")
    df = get_stall_training_data()

    print(f"🔥 컬럼 목록: {df.columns.tolist()}")
    print(df.head())

    if df.empty:
        print("⚠️ 학습 데이터 없음")
        return

    # --------------------------------------------------------
    # 1) user_id 컬럼 표준화 (customer_idx → user_id)
    # --------------------------------------------------------
    if "user_id" not in df.columns:
        if "customer_idx" in df.columns:
            print("🔄 customer_idx → user_id 로 자동 변경")
            df["user_id"] = df["customer_idx"]
        else:
            raise KeyError("❌ customer_idx / user_id 컬럼이 없습니다.")

    # --------------------------------------------------------
    # 2) store id 자동 탐지
    # --------------------------------------------------------
    store_col = find_store_id_column(df)
    print(f"★ Detected store id column: {store_col}")

    # --------------------------------------------------------
    # 3) label 자동 생성
    # --------------------------------------------------------
    df = generate_label(df)

    # --------------------------------------------------------
    # 4) user/store PK null 제거
    # --------------------------------------------------------
    df = df.dropna(subset=["user_id", store_col, "label"])
    df["user_id"] = df["user_id"].astype(int)
    df[store_col] = df[store_col].astype(int)
    df["label"] = df["label"].astype(float)  # BCELoss expects float

    # --------------------------------------------------------
    # 5) user/store idx 매핑
    # --------------------------------------------------------
    unique_users = sorted(df["user_id"].unique())
    user2idx = {int(u): i for i, u in enumerate(unique_users)}

    unique_stores = sorted(df[store_col].unique())
    store2idx = {int(s): i for i, s in enumerate(unique_stores)}

    df["user_idx"] = df["user_id"].map(user2idx)
    df["store_idx_mapped"] = df[store_col].map(store2idx)

    num_users = len(user2idx)
    num_stores = len(store2idx)

    print(f"num_users = {num_users}, num_stores = {num_stores}")
    print("user2idx 예:", list(user2idx.items())[:5])
    print("store2idx 예:", list(store2idx.items())[:5])

    # --------------------------------------------------------
    # 6) Dataset / DataLoader
    # --------------------------------------------------------
    dataset = StallTrainDataset(df, user_col="user_idx", store_col="store_idx_mapped")
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # --------------------------------------------------------
    # 7) Model
    # --------------------------------------------------------
    model = StallRecommender(num_users=num_users, num_stores=num_stores)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --------------------------------------------------------
    # 8) Device
    # --------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"🚀 Using device: {device}")

    # --------------------------------------------------------
    # 9) Training Loop
    # --------------------------------------------------------
    print("🚀 학습 시작…")
    for epoch in range(5):
        model.train()
        total_loss = 0.0

        for batch in loader:
            # 텐서를 GPU로 이동
            for k in batch:
                batch[k] = batch[k].to(device)

            preds = model(batch)  # (B,)
            loss = criterion(preds, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📌 Epoch {epoch+1}/5 | Loss: {total_loss:.4f}")

    # --------------------------------------------------------
    # 10) 체크포인트 저장
    # --------------------------------------------------------
    ckpt = {
        "state_dict": model.state_dict(),
        "user2idx": user2idx,
        "store2idx": store2idx,
    }
    torch.save(ckpt, "stall_recommender.pt")
    print("🎉 모델 & 매핑이 저장되었습니다 → stall_recommender.pt")


if __name__ == "__main__":
    main()