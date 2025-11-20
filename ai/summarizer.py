# summarizer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 예시 모델 (원하는 KoBART 요약 모델로 교체 가능)
SUM_MODEL_NAME = "gogamza/kobart-base-v2"  # 또는 요약에 특화된 KoBART 모델

class ReviewSummarizer:
    def __init__(self, model_name: str = SUM_MODEL_NAME, max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_length = max_length

    def summarize(self, text: str, summary_len: int = 48):
        inputs = self.tokenizer(
            [text],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=summary_len,
                num_beams=4,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary