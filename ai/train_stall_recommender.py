import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from ai.app.models.ml.stall_recommender import StallRecommender

# DB 대신 CSV/임시데이터 읽는 구조로 가정
from app.services.review_service import get_all_reviews_for_training

MODEL_PATH = "models/stall_recommender.pt"

def train():
    print("📥 Loading review data...")
    df = get_all_reviews_for_training()   # store_idx, review_text, taste, price, kindness

    if df.shape[0] < 10:
        print("❌ 리뷰데이터가 부족하여 학습 불가")
        return

    print("📌 TF-IDF Vectorizing...")

    vectorizer = TfidfVectorizer(
        max_features=300,   # 입력 차원 고정
        min_df=1            # 데이터 적어도 통과
    )
    X = vectorizer.fit_transform(df["review_text"]).toarray()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_taste = torch.tensor(df["taste"].values, dtype=torch.float32).view(-1, 1)
    y_price = torch.tensor(df["price"].values, dtype=torch.float32).view(-1, 1)
    y_kindness = torch.tensor(df["kindness"].values, dtype=torch.float32).view(-1, 1)

    model = StallRecommender(input_dim=X_tensor.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("🔥 Training start...")

    for epoch in range(20):
        optimizer.zero_grad()
        taste_pred, price_pred, kindness_pred = model(X_tensor)

        loss = (
            criterion(taste_pred, y_taste) +
            criterion(price_pred, y_price) +
            criterion(kindness_pred, y_kindness)
        )

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/20 | Loss: {loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vectorizer": vectorizer
    }, MODEL_PATH)

    print(f"✅ Training completed. Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()