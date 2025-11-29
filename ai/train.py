# train.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from typing import List, Dict

from app.db.mysql import get_reviews_by_store
from app.models.multitask_model import MultiTaskReviewModel, MODEL_NAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "multitask_review_model.pt"


# 🟦 리뷰 전체 가져오는 함수 (상점별 말고 전체)
def get_all_reviews() -> List[Dict]:
    """
    DB의 store_reviews 테이블의 모든 리뷰와 별점을 가져옴.
    감성(sentiment)과 독성(toxicity)은 아직 없기 때문에
    별점 기반 약한 라벨링을 사용.
    """
    import pymysql
    from app.db.mysql import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_CHARSET

    conn = pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset=MYSQL_CHARSET,
        cursorclass=pymysql.cursors.DictCursor,
    )

    with conn.cursor() as cur:
        cur.execute("""
            SELECT review_text, rating
            FROM store_reviews
            WHERE review_text IS NOT NULL
              AND rating IS NOT NULL
              AND is_blocked = 0
        """)
        rows = cur.fetchall()

    conn.close()
    return rows


# 🟦 약한 라벨링 (weak labeling)
def weak_sentiment_label(rating: int) -> int:
    """
    별점 기반 감성 라벨 자동 생성
    1~2 → negative
    3 → neutral
    4~5 → positive
    """
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


def weak_toxicity_label(rating: int) -> int:
    """
    별점 기반 독성 약한 라벨 (선택 사항)
    별점 낮으면 toxic 가능성 높다고 가정
    """
    return 1 if rating == 1 else 0


# 🟦 PyTorch Dataset
class ReviewDataset(Dataset):
    def __init__(self, max_length=128):
        self.data = get_all_reviews()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.max_length = max_length

        print(f"📌 Loaded {len(self.data)} reviews for training.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = str(row["review_text"])
        rating = int(row["rating"])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),

            "rating": torch.tensor(rating - 1, dtype=torch.long),          # 0~4
            "sentiment": torch.tensor(weak_sentiment_label(rating)),      # 0~2
            "toxicity": torch.tensor(weak_toxicity_label(rating))         # 0~1
        }


# 🟦 학습 함수
def train(
    epochs=3,
    batch_size=8,
    lr=2e-5
):
    print("🔥 Initializing dataset...")
    dataset = ReviewDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("🔥 Loading model...")
    model = MultiTaskReviewModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    loss_fn_rating = nn.CrossEntropyLoss()
    loss_fn_sentiment = nn.CrossEntropyLoss()
    loss_fn_toxicity = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            rating = batch["rating"].to(DEVICE)
            sentiment = batch["sentiment"].to(DEVICE)
            toxicity = batch["toxicity"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=mask)

            loss_rating = loss_fn_rating(outputs["rating_logits"], rating)
            loss_sentiment = loss_fn_sentiment(outputs["sentiment_logits"], sentiment)
            loss_toxicity = loss_fn_toxicity(outputs["toxicity_logits"], toxicity)

            loss = loss_rating + loss_sentiment + loss_toxicity

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📘 Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    print("💾 Saving model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("✅ Model saved as multitask_review_model.pt")


if __name__ == "__main__":
    train()