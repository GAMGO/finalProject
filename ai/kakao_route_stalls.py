from dotenv import load_dotenv
import os
import math
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

load_dotenv()
# ==============================
# 1) Kakao Directions API 호출
# ==============================
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")

DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"


def get_route_from_kakao(
    start: Dict[str, float],
    destination: Dict[str, float],
    waypoints: Optional[List[Dict[str, float]]] = None,
    priority: str = "TIME",  # or "DISTANCE"
) -> Dict:
    """
    Kakao 모빌리티 Directions API 호출해서
    - summary (duration, distance 등)
    - path: [ (lat, lng), ... ] 형태의 polyline 좌표 리스트
    를 반환.

    start, destination, waypoints 형식:
    {
      "lat": 37.5665,
      "lng": 126.9780
    }
    """

    if KAKAO_REST_API_KEY is None:
        raise RuntimeError("환경변수 KAKAO_REST_API_KEY 가 설정되어 있지 않습니다.")

    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }

    # 경유지 문자열 만들기: "lng1,lat1|lng2,lat2" 형식
    if waypoints:
        waypoints_param = "|".join(
            f"{wp['lng']},{wp['lat']}" for wp in waypoints
        )
    else:
        waypoints_param = ""

    params = {
        "origin": f"{start['lng']},{start['lat']}",
        "destination": f"{destination['lng']},{destination['lat']}",
        "priority": priority,  # "TIME" or "DISTANCE"
    }
    if waypoints_param:
        params["waypoints"] = waypoints_param

    resp = requests.get(DIRECTIONS_URL, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    routes = data.get("routes", [])
    if not routes:
        raise RuntimeError("경로가 없습니다. Kakao API 응답 routes가 비어 있음.")

    route0 = routes[0]
    summary = route0.get("summary", {})
    sections = route0.get("sections", [])

    # sections[*].roads[*].vertexes: [lng1, lat1, lng2, lat2, ...]
    path: List[Tuple[float, float]] = []

    for section in sections:
        for road in section.get("roads", []):
            v = road.get("vertexes", [])
            # 2개씩 끊어서 (lng, lat)
            for i in range(0, len(v), 2):
                lng = v[i]
                lat = v[i + 1]
                path.append((lat, lng))  # (lat, lng) 순서로 저장

    result = {
        "summary": summary,
        "path": path,  # [(lat, lng), ...]
    }
    return result


# ==============================
# 2) 거리 계산 유틸 (haversine)
# ==============================
def haversine(lat1, lon1, lat2, lon2):
    """
    위경도 두 점 사이 거리(m)를 반환
    """
    R = 6371000  # 지구 반지름 (m)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def min_distance_to_route(
    stall_lat: float, stall_lng: float, route_points: List[Tuple[float, float]]
) -> float:
    """
    노점 좌표와 경로 polyline(점 리스트) 사이의 '최소 거리(m)'를 계산.
    간단하게: polyline의 각 점과 노점 사이의 haversine 거리 중 최소값 사용.
    (정확한 선분-점 거리보다는 단순하지만 MVP엔 충분)
    """
    if not route_points:
        return float("inf")

    dists = [
        haversine(stall_lat, stall_lng, lat, lng)
        for (lat, lng) in route_points
    ]
    return min(dists)


# ==============================
# 3) 경로 주변 노점 필터링
# ==============================
def filter_stalls_near_route(
    stalls_df: pd.DataFrame,
    route_points: List[Tuple[float, float]],
    radius_m: float = 300.0,
) -> pd.DataFrame:
    """
    stalls_df: 최소한 ["stall_id", "lat", "lng"] 컬럼 포함
    route_points: [(lat, lng), ...] 형식
    radius_m: 경로 기준 허용 반경(m)
    """
    dists = []
    for _, row in stalls_df.iterrows():
        d = min_distance_to_route(row["lat"], row["lng"], route_points)
        dists.append(d)

    stalls_df = stalls_df.copy()
    stalls_df["distance_to_route_m"] = dists

    near_df = stalls_df[stalls_df["distance_to_route_m"] <= radius_m].sort_values(
        "distance_to_route_m"
    )
    return near_df


# ==============================
# 4) 데모용 더미 노점 데이터 생성
#    (실제에선 DB에서 SELECT 해서 df로 만들면 됨)
# ==============================
def build_dummy_stalls(center_lat: float, center_lng: float, n: int = 50) -> pd.DataFrame:
    """
    중심 좌표 주변에 대충 랜덤 노점 n개 생성.
    실서비스에서는 여기 대신 DB에서 가져오면 됨.
    """
    np.random.seed(42)
    # 위도/경도 약 ±0.01 도 정도 랜덤 → 대충 1km 근방
    lats = center_lat + (np.random.rand(n) - 0.5) * 0.02
    lngs = center_lng + (np.random.rand(n) - 0.5) * 0.02

    data = []
    for i in range(n):
        data.append(
            {
                "stall_id": i + 1,
                "name": f"노점_{i+1}",
                "lat": lats[i],
                "lng": lngs[i],
            }
        )

    df = pd.DataFrame(data)
    return df


# ==============================
# 5) 엔드투엔드 실행 예시
# ==============================
if __name__ == "__main__":
    # 예시: 서울 시청 → 강남역, 중간에 고속터미널 근처를 경유지로 가정
    start = {"lat": 37.5665, "lng": 126.9780}      # 서울시청 근처
    destination = {"lat": 37.4979, "lng": 127.0276}  # 강남역 근처
    waypoints = [
        {"lat": 37.5048, "lng": 127.0041},  # 고속터미널 근처 (예시)
    ]

    print("🔹 Kakao Directions API 호출 중...")
    route = get_route_from_kakao(start, destination, waypoints=waypoints, priority="TIME")
    path = route["path"]
    summary = route["summary"]

    print(f"총 거리: {summary.get('distance', 'N/A')} m")
    print(f"총 시간: {summary.get('duration', 'N/A')} sec")
    print(f"경로 좌표 개수: {len(path)}")

    # 노점 데이터 (실제에선 DB에서 불러오기)
    # 경로 중간 지점 근처를 중심으로 더미 노점 생성
    mid_lat, mid_lng = path[len(path) // 2]
    stalls_df = build_dummy_stalls(mid_lat, mid_lng, n=100)

    print(f"노점 전체 개수: {len(stalls_df)}")

    # 경로 300m 이내 노점만 필터링
    near_stalls = filter_stalls_near_route(stalls_df, path, radius_m=300.0)

    print(f"경로 300m 이내 노점 개수: {len(near_stalls)}")
    print(near_stalls.head(10)[["stall_id", "name", "lat", "lng", "distance_to_route_m"]])

    # 🔻 여기까지가 "경로 → 주변 노점 필터링"
    # 🔻 이 near_stalls DataFrame을 너가 만든 PyTorch 추천/추첨 모델에 넣으면 됨.
    #
    # 예)
    # 1) near_stalls 에 distance_from_route = distance_to_route_m 등 피처 추가
    # 2) user_id, 시간정보, 평점/리뷰 정보 붙이기
    # 3) 이전에 만든 recommend(model, candidates_df, user_id, ...) 호출