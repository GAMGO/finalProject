from fastapi import FastAPI
from app.routes.review_summary import router as summary_router
from app.routes.recommend_route import router as route_recommend_router

app = FastAPI()

app.include_router(summary_router, prefix="/api")
app.include_router(route_recommend_router, prefix="/recommend")
@app.get("/")
def root():
    return {"status": "ok"}