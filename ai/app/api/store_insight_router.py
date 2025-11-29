from fastapi import APIRouter
from pydantic import BaseModel
from app.db.mysql import get_reviews_by_store
from app.models.analyzer import ReviewAnalyzer
from app.models.summarizer import ReviewSummarizer

router = APIRouter(prefix="/api/store")

analyzer = ReviewAnalyzer()
summarizer = ReviewSummarizer()

class StoreRequest(BaseModel):
    store_id: int
    store_name: str
    lat: float
    lng: float

@router.post("/analysis")
def analyze_store(req: StoreRequest):
    store_id = req.store_id
    store_name = req.store_name

    reviews = get_reviews_by_store(store_id)
    if not reviews:
        return {
            "store_id": store_id,
            "store_name": store_name,
            "review_count": 0,
            "message": "No reviews"
        }

    preds = []
    fulltext = ""
    rating_dist = [0,0,0,0,0]

    for r in reviews:
        text = r["review_text"]
        fulltext += text + ". "

        pred = analyzer.analyze(text)
        preds.append(pred)

        rating_dist[pred["rating_pred"] - 1] += 1

    sentiment_summary = {
        "positive": sum(p["sentiment"] == 2 for p in preds),
        "neutral": sum(p["sentiment"] == 1 for p in preds),
        "negative": sum(p["sentiment"] == 0 for p in preds),
    }

    toxicity_summary = {
        "clean": sum(p["toxicity"] == 0 for p in preds),
        "toxic": sum(p["toxicity"] == 1 for p in preds),
    }

    # 리뷰 전체 요약
    summary = summarizer.summarize(fulltext, max_sentences=3)

    return {
        "store_id": store_id,
        "store_name": store_name,
        "review_count": len(reviews),
        "rating": {
            "distribution": rating_dist,
            "avg": sum((i+1)*rating_dist[i] for i in range(5)) / len(reviews)
        },
        "sentiment": sentiment_summary,
        "toxicity": toxicity_summary,
        "summary": summary
    }