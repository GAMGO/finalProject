import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "klue/bert-base"


class MultiTaskReviewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.2)

        self.rating_head = nn.Linear(hidden, 5)
        self.sentiment_head = nn.Linear(hidden, 3)
        self.toxicity_head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])

        return {
            "rating_logits": self.rating_head(cls),
            "sentiment_logits": self.sentiment_head(cls),
            "toxicity_logits": self.toxicity_head(cls),
        }