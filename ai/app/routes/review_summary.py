# app/routers/store_summary_router.py

from fastapi import APIRouter
from app.services.review_service import get_reviews
from app.services.summarizer import summarize_reviews

router = APIRouter()

@router.get("/stores/{store_idx}/summary")
def get_summary(store_idx: int):

    reviews = get_reviews(store_idx)

    summary = summarize_reviews(reviews)

    return {
        "store_id": store_idx,
        "review_count": len(reviews),
        "summary": summary
    }