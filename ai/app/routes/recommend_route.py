from fastapi import APIRouter
from app.services.recommend_service import recommend_near_point
from app.repositories.store_repository import get_stores_near_location
from app.services.recommend_service import strip_reason_for_unrecommended  # 🔥 추가
from app.services.llm_reason_service import generate_recommend_reason

router = APIRouter()
_llm_cache = {}

router = APIRouter()
_llm_cache = {}


# -------------------------------------------------------------------
# 캐시 기반 추천 결과 호출
# -------------------------------------------------------------------
def get_recommend_for_point(user_id, stores, lat, lng, point_type):
    key = (round(lat, 6), round(lng, 6), point_type)

    if key in _llm_cache:
        return _llm_cache[key]

    result = recommend_near_point(
        user_id=user_id,
        stores=stores,
        lat=lat,
        lng=lng,
        point_type=point_type,
        limit=5
    )

    _llm_cache[key] = result
    return result


# -------------------------------------------------------------------
# rating ≥ 3.0 AND recommended=True 일 때만 reason 생성
# -------------------------------------------------------------------
def attach_reason_filter(stores, origin):
    """
    stores: 추천된 가게들의 리스트
    origin: "출발지" / "경유지" / "도착지"
    """
    results = []

    for s in stores:
        rating = 0.0
        try:
            rating = float(s.get("rating", 0.0))
        except:
            rating = 0.0

        recommended = bool(s.get("recommended", False))

        # 조건 충족 시 reason 생성
        if rating >= 3.0 and recommended:
            s["reason"] = generate_recommend_reason(s, origin)

        results.append(s)

    return results


# -------------------------------------------------------------------
# 메인 추천 엔드포인트
# -------------------------------------------------------------------
@router.post("/route")
def recommend_route(payload: dict):
    user_id = int(payload.get("user_id", 0))
    start = payload["start"]
    waypoints = payload.get("waypoints") or []
    end = payload["end"]

    used_ids = set()

    # ------------------------------
    # 출발지
    # ------------------------------
    start_lat = float(start["lat"])
    start_lng = float(start["lng"])

    start_stores = get_stores_near_location(start_lat, start_lng)
    start_reco = get_recommend_for_point(user_id, start_stores, start_lat, start_lng, "출발지")

    # reason 필터 적용
    start_reco = attach_reason_filter(start_reco, "출발지")

    for s in start_reco:
        used_ids.add(s["idx"])

    # ------------------------------
    # 경유지
    # ------------------------------
    waypoint_recos = []
    for wp in waypoints:
        wp_lat = float(wp["lat"])
        wp_lng = float(wp["lng"])

        wp_stores = get_stores_near_location(wp_lat, wp_lng)
        wp_reco = get_recommend_for_point(user_id, wp_stores, wp_lat, wp_lng, "경유지")

        # 중복 제거
        filtered = [s for s in wp_reco if s["idx"] not in used_ids]

        # reason 필터 적용
        filtered = attach_reason_filter(filtered, "경유지")

        for s in filtered:
            used_ids.add(s["idx"])

        waypoint_recos.append(filtered)

    # ------------------------------
    # 도착지
    # ------------------------------
    end_lat = float(end["lat"])
    end_lng = float(end["lng"])

    end_stores = get_stores_near_location(end_lat, end_lng)
    end_reco = get_recommend_for_point(user_id, end_stores, end_lat, end_lng, "도착지")

    # 중복 제거
    end_reco = [s for s in end_reco if s["idx"] not in used_ids]

    # reason 적용
    end_reco = attach_reason_filter(end_reco, "도착지")

    return {
        "start": start_reco,
        "waypoints": waypoint_recos,
        "end": end_reco
    }