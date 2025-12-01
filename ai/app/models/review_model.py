from pydantic import BaseModel
from typing import Optional


class Review:
    def __init__(self, review_text, rating):
        self.review_text = review_text
        self.rating = rating

    @staticmethod
    def from_row(row):
        return Review(
            review_text=row.get("review_text"),
            rating=row.get("rating")
        )