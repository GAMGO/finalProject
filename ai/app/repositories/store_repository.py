# app/repositories/store_repository.py
import math
import pymysql
from app.config.database import get_db_connection


# Haversine 거리 계산 (meters)
def calc_distance(lat1, lng1, lat2, lng2):
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lng2 - lng1)

    a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lam/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def get_stores_near_location(lat: float, lng: float, radius_m: int = 2000):
    """
    특정 좌표 근처의 노점 리스트 조회
    - 기본 반경 = 2000m (2km)
    - 필수 필드: idx, lat, lng, sentiment_score, rating, distance, hour_sin
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1차 후보: 좌표 박스 필터링 (속도)
            lat_delta = radius_m / 111320
            lng_delta = radius_m / (111320 * math.cos(math.radians(lat)))

            min_lat = lat - lat_delta
            max_lat = lat + lat_delta
            min_lng = lng - lng_delta
            max_lng = lng + lng_delta

            cur.execute(
                """
                SELECT
                    s.IDX AS store_id,
                    s.lat AS lat,
                    s.lng AS lng,
                    IFNULL(r.avg_sentiment, 0) AS sentiment_score,
                    IFNULL(r.avg_rating, 0) AS rating,
                    IFNULL(r.hour_sin, 0) AS hour_sin
                FROM store s
                LEFT JOIN (
                    SELECT
                        store_idx,
                        AVG(sentiment_score) AS avg_sentiment,
                        AVG(rating) AS avg_rating,
                        AVG(SIN(HOUR(created_at)/3.14)) AS hour_sin
                    FROM store_reviews
                    WHERE is_blocked = 0
                    GROUP BY store_idx
                ) r ON s.IDX = r.store_idx
                WHERE s.lat BETWEEN %s AND %s
                AND s.lng BETWEEN %s AND %s
                """,
                (min_lat, max_lat, min_lng, max_lng)
            )

            rows = cur.fetchall()

        # 2차 후보: 실제 haversine 거리 계산
        results = []
        for row in rows:
            dist = calc_distance(lat, lng, row["lat"], row["lng"])
            if dist <= radius_m:
                results.append({
                    "idx": row["store_id"],
                    "lat": row["lat"],
                    "lng": row["lng"],
                    "sentiment_score": float(row["sentiment_score"]),
                    "rating": float(row["rating"]),
                    "hour_sin": float(row["hour_sin"]),
                    "distance": float(dist)
                })

        return results

    finally:
        conn.close()