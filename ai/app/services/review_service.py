# app/services/review_service.py
from app.config.database import get_db_connection

def get_reviews(store_idx: int):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT review_text, rating
                FROM store_reviews
                WHERE store_idx = %s AND is_blocked = 0
                ORDER BY created_at DESC
            """, (store_idx,))
            return cur.fetchall()
    finally:
        conn.close()

import pandas as pd
from app.config.database import get_db_connection

def get_all_reviews_for_training():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT store_idx, review_text, rating
                FROM store_reviews
                WHERE is_blocked = 0
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    # rows → pandas DataFrame 변환
    df = pd.DataFrame(rows, columns=["store_idx", "review_text", "rating"])
    return df