 #ai/train_stall_recommender.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from app.repositories.stall_training_repository import get_stall_training_data
from datasets.stall_dataset import StallTrainDataset
from app.models.stall_recommender import StallRecommender


# --------------------------------------------------------
# store_id 컬럼 자동 탐지
# --------------------------------------------------------
def find_store_id_column(df):
    """DataFrame에서 store id 컬럼을 자동으로 탐지한다."""
    candidates = ["store_idx", "store_id", "storeId", "idx",
                  "STORE_IDX", "STORE_ID", "STOREID"]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError("❌ DataFrame에 store id 컬럼이 없습니다.")


# --------------------------------------------------------
# 학습 메인 함수
# --------------------------------------------------------
def main():
    print("📥 학습 데이터 로드...")
    df = get_stall_training_data()
    print("🔥 DF columns:", df.columns.tolist())
    print(df.head())
    if df.empty:
        print("⚠️ 학습 데이터가 없습니다.")
        return

    # --- store idx 자동 탐지 ---
    store_col = find_store_id_column(df)
    print(f"★ Detected store id column: {store_col}")

    # --- user/store PK null 제거 + int 캐스팅 ---
    df = df.dropna(subset=["user_id", store_col])
    df["user_id"] = df["user_id"].astype(int)
    df[store_col] = df[store_col].astype(int)

    # --------------------------------------------------------
    # 1) PK → 연속 인덱스로 매핑 (user2idx, store2idx)
    # --------------------------------------------------------
    unique_users = sorted(df["user_id"].unique())
    user2idx = {int(u): i for i, u in enumerate(unique_users)}

    unique_stores = sorted(df[store_col].unique())
    store2idx = {int(s): i for i, s in enumerate(unique_stores)}

    df["user_idx"] = df["user_id"].map(user2idx)
    df["store_idx_mapped"] = df[store_col].map(store2idx)

    num_users = len(user2idx)
    num_stores = len(store2idx)

    print(f"num_users={num_users}, num_stores={num_stores}")
    print("예시 매핑 user2idx:", list(user2idx.items())[:5])
    print("예시 매핑 store2idx:", list(store2idx.items())[:5])

    # --- Dataset / DataLoader ---
    dataset = StallTrainDataset(df, user_col="user_idx", store_col="store_idx_mapped")
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # --- Model ---
    model = StallRecommender(num_users=num_users, num_stores=num_stores)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"🚀 Using device: {device}")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    print("🚀 학습 시작...")
    for epoch in range(5):
        model.train()
        total_loss = 0.0

        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            preds = model(batch)              # (B,)
            loss = criterion(preds, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📌 Epoch {epoch+1}/5 | Loss: {total_loss:.4f}")

    # --------------------------------------------------------
    # 2) state_dict + 매핑을 같이 저장
    # --------------------------------------------------------
    ckpt = {
        "state_dict": model.state_dict(),
        "user2idx": user2idx,
        "store2idx": store2idx,
    }
    torch.save(ckpt, "stall_recommender.pt")
    print("🎉 모델 & 매핑 저장 완료 → stall_recommender.pt")


if __name__ == "__main__":
    main()