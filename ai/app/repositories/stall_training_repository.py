# app/repositories/stall_training_repository.py
import pandas as pd
from app.config.database import get_db_connection


def get_stall_training_data() -> pd.DataFrame:
    """
    store + store_reviews 기반 학습 데이터 구성
    store_idx 포함 필수!
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    r.customer_idx AS user_id,
                    r.store_idx AS store_idx,              -- ⭐ 반드시 포함
                    s.lat AS latitude,
                    s.lng AS longitude,
                    0 AS distance_from_route,              -- 학습 중에는 0
                    r.sentiment_score AS sentiment_score,
                    r.rating AS rating,
                    SIN(HOUR(r.created_at) / 3.14) AS hour_sin,
                    CASE WHEN r.rating >= 4 THEN 1 ELSE 0 END AS label
                FROM store_reviews r
                JOIN store s ON r.store_idx = s.IDX
                WHERE r.is_blocked = 0
                  AND r.review_text IS NOT NULL
                  AND r.review_text != ''
                  AND r.sentiment_score IS NOT NULL
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "store_idx",              # ⭐ 추가됨
            "latitude",
            "longitude",
            "distance_from_route",
            "sentiment_score",
            "rating",
            "hour_sin",
            "label",
        ],
    )
    return df