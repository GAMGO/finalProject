# ai/train_stall_recommender.py
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

    # --- user/store embedding 크기 계산 ---
    num_users = int(df["user_id"].max()) + 1
    num_stores = int(df[store_col].max()) + 1

    print(f"num_users={num_users}, num_stores={num_stores}")

    # --- Dataset / DataLoader ---
    dataset = StallTrainDataset(df, store_col)
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

    torch.save(model.state_dict(), "stall_recommender.pt")
    print("🎉 모델 저장 완료 → stall_recommender.pt")


if __name__ == "__main__":
    main()