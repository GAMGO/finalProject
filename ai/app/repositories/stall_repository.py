# ai/repositories/stall_repository.py
from app.config.database import get_db_connection

def get_nearby_stores(lat, lng, radius_km=3):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    idx, store_name, lat, lng, food_type, rating
                FROM store
                WHERE lat BETWEEN %s - 0.03 AND %s + 0.03
                  AND lng BETWEEN %s - 0.03 AND %s + 0.03
            """, (lat, lat, lng, lng))
            return cur.fetchall()
    finally:
        conn.close()