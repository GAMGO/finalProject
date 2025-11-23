import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

MODEL_NAME = "klue/bert-base"

class ReviewDataset(Dataset):
    def __init__(self, csv_path, max_length=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["review_text"])
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "rating": torch.tensor(int(row["rating"]) - 1, dtype=torch.long),  # 0~4
            "sentiment": torch.tensor(int(row["sentiment"]), dtype=torch.long),
            "toxicity": torch.tensor(int(row["toxicity"]), dtype=torch.long),
        }
        return item