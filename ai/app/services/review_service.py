# app/services/review_service.py
from app.config.database import get_db_connection
from app.models.review_model import Review
import pandas as pd

def get_reviews(store_idx: int) -> list[Review]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT review_text, rating, sentiment_score, sentiment_label
                FROM store_reviews
                WHERE store_idx = %s AND is_blocked = 0
                ORDER BY created_at DESC
            """, (store_idx,))
            rows = cur.fetchall()

            return [Review.from_row(row) for row in rows]

    finally:
        conn.close()


def get_all_reviews_for_training() -> pd.DataFrame:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT store_idx, review_text, rating
                FROM store_reviews
                WHERE is_blocked = 0
            """)
            rows = cur.fetchall()

            df = pd.DataFrame(rows, columns=["store_idx", "review_text", "rating"])
            return df

    finally:
        conn.close()