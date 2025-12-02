from fastapi import FastAPI
from app.routes.review_summary import router as summary_router
from app.routes.recommend_route import router as route_recommend_router

app = FastAPI()

app.include_router(summary_router, prefix="/api")
app.include_router(route_recommend_router, prefix="/recommend")
@app.get("/")
def root():
    return {"status": "ok"}

# ✅ Spring이 통계 갱신 요청할 엔드포인트
# @app.post("/reviews/recompute")
# def recompute(store_idx: int = Query(..., description="가게 ID")):
#     print(f"[FastAPI] Recomputing review stats for store {store_idx}")
#     return {"ok": True, "store_idx": store_idx}