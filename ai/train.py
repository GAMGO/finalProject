# train.py
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from dataset import ReviewDataset
from model import MultiTaskReviewModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    dataset = ReviewDataset("reviews.csv", max_length=128)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = MultiTaskReviewModel().to(DEVICE)

    criterion_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    num_epochs = 3
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            rating_labels = batch["rating"].to(DEVICE)
            sentiment_labels = batch["sentiment"].to(DEVICE)
            toxicity_labels = batch["toxicity"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss_rating = criterion_ce(outputs["rating_logits"], rating_labels)
            loss_sentiment = criterion_ce(outputs["sentiment_logits"], sentiment_labels)
            loss_toxicity = criterion_ce(outputs["toxicity_logits"], toxicity_labels)

            # 가중치는 필요에 따라 조정
            loss = loss_rating + loss_sentiment + loss_toxicity

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ===== validation =====
        model.eval()
        val_loss = 0.0
        correct_rating = 0
        total_rating = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                rating_labels = batch["rating"].to(DEVICE)
                sentiment_labels = batch["sentiment"].to(DEVICE)
                toxicity_labels = batch["toxicity"].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                loss_rating = criterion_ce(outputs["rating_logits"], rating_labels)
                loss_sentiment = criterion_ce(outputs["sentiment_logits"], sentiment_labels)
                loss_toxicity = criterion_ce(outputs["toxicity_logits"], toxicity_labels)
                loss = loss_rating + loss_sentiment + loss_toxicity
                val_loss += loss.item()

                pred_rating = outputs["rating_logits"].argmax(dim=-1)
                correct_rating += (pred_rating == rating_labels).sum().item()
                total_rating += rating_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        rating_acc = correct_rating / total_rating if total_rating > 0 else 0.0

        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Rating Acc: {rating_acc:.4f}"
        )

    torch.save(model.state_dict(), "multitask_review_model.pt")
    print("✅ Model saved to multitask_review_model.pt")

if __name__ == "__main__":
    train()