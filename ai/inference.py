# inference.py
import torch
from transformers import AutoTokenizer
from model import MultiTaskReviewModel, MODEL_NAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ReviewAnalyzer:
    def __init__(self, model_path="multitask_review_model.pt", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = MultiTaskReviewModel().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        self.max_length = max_length

    def predict(self, text: str):
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        rating_idx = outputs["rating_logits"].argmax(dim=-1).item()
        sentiment_idx = outputs["sentiment_logits"].argmax(dim=-1).item()
        toxicity_idx = outputs["toxicity_logits"].argmax(dim=-1).item()

        rating = rating_idx + 1  # 1~5
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        toxicity_map = {0: "clean", 1: "toxic"}

        return {
            "rating": rating,
            "sentiment": sentiment_map.get(sentiment_idx, "unknown"),
            "toxicity": toxicity_map.get(toxicity_idx, "unknown"),
        }