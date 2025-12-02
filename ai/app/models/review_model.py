from pydantic import BaseModel
from typing import Optional, Any

class Review(BaseModel):
    review_text: str
    rating: int
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None

    @staticmethod
    def from_row(row: dict[str, Any]):
        return Review(
            review_text=row.get("review_text"),
            rating=row.get("rating"),
            sentiment_score=row.get("sentiment_score"),
            sentiment_label=row.get("sentiment_label")
        )