from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.review_summary import router as summary_router
from app.routes.recommend_route import router as route_recommend_router

app = FastAPI()

# ✅ 프론트 도메인들 (Vite dev 서버 주소 넣기)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    # 필요하면 더 추가
    # "http://localhost:3000",
]

# ✅ CORS 미들웨어 등록 (라우터 include보다 위에서)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 개발 중에는 ["*"] 해도 됨
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, OPTIONS 등 전부 허용
    allow_headers=["*"],        # Content-Type, Authorization 등
)

# ✅ 라우터 등록
app.include_router(summary_router, prefix="/api")
app.include_router(route_recommend_router, prefix="/recommend")


@app.get("/")
def root():
  return {"status": "ok"}

# ✅ Spring이 통계 갱신 요청할 엔드포인트 (주석 유지)
# @app.post("/reviews/recompute")
# def recompute(store_idx: int = Query(..., description="가게 ID")):
#     print(f"[FastAPI] Recomputing review stats for store {store_idx}")
#     return {"ok": True, "store_idx": store_idx}
