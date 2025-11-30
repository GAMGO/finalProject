# app/repositories/stall_repository.py
from app.config.database import get_db_connection

def get_all_stalls():
    """
    stalls 테이블 전체 가져오기
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT idx, name, lat, lng, stall_category_id,
                       rating_avg, rating_count_log,
                       sentiment_score, price_level, hour_sin,
                       user_id
                FROM stalls
            """)
            rows = cur.fetchall()
            return rows
    finally:
        conn.close()
