# model.py
import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "klue/bert-base"

class MultiTaskReviewModel(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_rating_classes: int = 5,
        num_sentiment_classes: int = 3,
        num_toxicity_classes: int = 2,
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)

        self.rating_head = nn.Linear(hidden_size, num_rating_classes)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_classes)
        self.toxicity_head = nn.Linear(hidden_size, num_toxicity_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # [CLS] 토큰
        x = self.dropout(cls)

        rating_logits = self.rating_head(x)
        sentiment_logits = self.sentiment_head(x)
        toxicity_logits = self.toxicity_head(x)

        return {
            "rating_logits": rating_logits,
            "sentiment_logits": sentiment_logits,
            "toxicity_logits": toxicity_logits,
        }