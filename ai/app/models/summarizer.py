import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from app.models.multitask_model import MODEL_NAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ReviewSummarizer:
    def __init__(self):
        print("🔄 Loading BERT Summarizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()
        print("✅ Summarizer Loaded")

    def split_sentences(self, text):
        t = text.replace("?", ".").replace("!", ".")
        return [s.strip() for s in t.split(".") if s.strip()]

    def embed(self, sentence):
        enc = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)
        return out.last_hidden_state[:, 0].cpu().numpy()[0]

    def summarize(self, text, max_sentences=2):
        sents = self.split_sentences(text)
        if len(sents) <= max_sentences:
            return text

        vectors = np.vstack([self.embed(s) for s in sents])
        scores = np.linalg.norm(vectors, axis=1)
        top_idx = scores.argsort()[::-1][:max_sentences]
        top_idx = sorted(top_idx)

        return ". ".join(sents[i] for i in top_idx)