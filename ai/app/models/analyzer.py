import torch
from transformers import AutoTokenizer

from app.models.multitask_model import MultiTaskReviewModel, MODEL_NAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "multitask_review_model.pt"  # ai 폴더에 위치

class ReviewAnalyzer:
    def __init__(self):
        print("🔄 Loading MultiTaskReviewModel...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = MultiTaskReviewModel().to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()
        print("✅ Analyzer Loaded")

    def analyze(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=mask)

        rating = out["rating_logits"].argmax(-1).item() + 1
        sentiment = out["sentiment_logits"].argmax(-1).item()
        toxicity = out["toxicity_logits"].argmax(-1).item()

        return {
            "rating_pred": rating,
            "sentiment": sentiment,
            "toxicity": toxicity
        }