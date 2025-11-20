# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from inference import ReviewAnalyzer
from summarizer import ReviewSummarizer

app = FastAPI(
    title="Review Analysis API",
    description="별점 예측 + 감성 + 독성 + 요약 API",
    version="0.1.0",
)

analyzer = ReviewAnalyzer(model_path="multitask_review_model.pt")
summarizer = ReviewSummarizer()

class ReviewRequest(BaseModel):
    text: str

class ReviewResponse(BaseModel):
    rating: int
    sentiment: str
    toxicity: str
    summary: str

@app.post("/analyze", response_model=ReviewResponse)
def analyze_review(req: ReviewRequest):
    analysis = analyzer.predict(req.text)
    summary = summarizer.summarize(req.text)

    return ReviewResponse(
        rating=analysis["rating"],
        sentiment=analysis["sentiment"],
        toxicity=analysis["toxicity"],
        summary=summary,
    )