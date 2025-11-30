# app/routes/stall_recommend_router.py
from fastapi import APIRouter
from app.pipelines.recommend_pipeline import recommend_pipeline

router = APIRouter()

@router.post("/stall/recommend")
def recommend_stall(payload: dict):
    start = payload["start"]
    dest = payload["dest"]
    via = payload.get("via", [])
    user_id = payload["user_id"]

    result = recommend_pipeline(start, dest, via, user_id)
    return {"recommendations": result}