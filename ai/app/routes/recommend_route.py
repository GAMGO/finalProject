from fastapi import APIRouter
from app.services.recommend_service import recommend_near_point
from app.repositories.store_repository import get_stores_near_location   # ⭐ 필요
from app.utils.distance import calculate_distance                        # ⭐ 필요

router = APIRouter()

@router.post("/route")
def recommend_route(payload: dict):
    user_id = int(payload.get("user_id", 0))
    start = payload.get("start")
    waypoints = payload.get("waypoints") or []
    end = payload.get("end")

    result = {}

    # --------------------------
    # 출발지 주변 가게 추천
    # --------------------------
    start_lat = float(start["lat"])
    start_lng = float(start["lng"])

    start_stores = get_stores_near_location(start_lat, start_lng)
    result["start_recommend"] = recommend_near_point(user_id, start_stores)

    # --------------------------
    # 경유지 주변 가게 추천
    # --------------------------
    waypoint_results = []
    for wp in waypoints:
        lat = float(wp["lat"])
        lng = float(wp["lng"])

        stores = get_stores_near_location(lat, lng)
        waypoint_results.append(recommend_near_point(user_id, stores))

    result["waypoint_recommend"] = waypoint_results

    # --------------------------
    # 도착지 주변 가게 추천
    # --------------------------
    end_lat = float(end["lat"])
    end_lng = float(end["lng"])

    end_stores = get_stores_near_location(end_lat, end_lng)
    result["end_recommend"] = recommend_near_point(user_id, end_stores)

    return result