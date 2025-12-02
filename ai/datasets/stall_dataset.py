# ai/datasets/stall_dataset.py
import torch
from torch.utils.data import Dataset

class StallTrainDataset(Dataset):
    def __init__(self, df, user_col: str, store_col: str):
        """
        df: DataFrame
        user_col: 연속 인덱스로 매핑된 user index 컬럼명
        store_col: 연속 인덱스로 매핑된 store index 컬럼명
        """

        # Long indices (이미 0 ~ N-1 로 매핑된 값)
        self.user_id = torch.tensor(df[user_col].values, dtype=torch.long)
        self.store_id = torch.tensor(df[store_col].values, dtype=torch.long)

        # Float features
        self.sentiment = torch.tensor(df["sentiment_score"].values, dtype=torch.float32)
        self.rating = torch.tensor(df["rating"].values, dtype=torch.float32)
        self.distance = torch.tensor(df["distance_from_route"].values, dtype=torch.float32)
        self.hour_sin = torch.tensor(df["hour_sin"].values, dtype=torch.float32)

        # Label (0 or 1)
        self.label = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_id[idx],
            "store_id": self.store_id[idx],
            "sentiment": self.sentiment[idx],
            "rating": self.rating[idx],
            "distance": self.distance[idx],
            "hour_sin": self.hour_sin[idx],
            "label": self.label[idx],
        }