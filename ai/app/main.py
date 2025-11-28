# app/main.py
from fastapi import FastAPI
from app.api.store_insight_router import router as store_router

app = FastAPI(title="Store Insight API")

app.include_router(store_router)