import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "klue/bert-base"

class ReviewSummarizer:
    def __init__(self, model_name: str = EMBED_MODEL):
        print("🔄 Loading Extractive Summarizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        print("✅ Summarizer Loaded")

    def sentence_split(self, text: str):
        # 단순 문장 분리
        s = text.replace("?", ".").replace("!", ".")
        return [x.strip() for x in s.split(".") if len(x.strip()) > 0]

    def embed(self, sentence: str):
        tokens = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        tokens = {k: v.to(DEVICE) for k,v in tokens.items()}

        with torch.no_grad():
            output = self.model(**tokens)
        cls = output.last_hidden_state[:, 0]
        return cls.cpu().numpy()[0]

    def summarize(self, text: str, max_sentences: int = 2):
        if not text or len(text.strip()) == 0:
            return ""

        sents = self.sentence_split(text)

        # 문장 하나면 그대로 반환
        if len(sents) <= max_sentences:
            return text

        # 문장 임베딩
        embeddings = np.vstack([self.embed(s) for s in sents])

        # 중요도 점수 (L2 norm)
        scores = np.linalg.norm(embeddings, axis=1)

        # 상위 문장 선택
        idxs = scores.argsort()[::-1][:max_sentences]
        idxs = sorted(idxs)  # 원래 순서 유지

        summary = ". ".join([sents[i] for i in idxs])
        return summary.strip()