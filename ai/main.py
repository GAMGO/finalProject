from fastapi import FastAPI
from app.routes.review_summary import router as summary_router
from app.routes.stall_recommend import router as stall_router

app = FastAPI()

app.include_router(summary_router, prefix="/api")
app.include_router(stall_router)
@app.get("/")
def root():
    return {"status": "ok"}