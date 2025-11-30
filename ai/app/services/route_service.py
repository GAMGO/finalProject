import os
import math
import requests

KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"

def get_route(start, dest, waypoints=None):
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}

    if waypoints:
        w = "|".join(f"{p['lng']},{p['lat']}" for p in waypoints)
    else:
        w = ""

    params = {
        "origin": f"{start['lng']},{start['lat']}",
        "destination": f"{dest['lng']},{dest['lat']}",
        "priority": "TIME",
    }
    if w:
        params["waypoints"] = w

    r = requests.get(DIRECTIONS_URL, headers=headers, params=params)
    r.raise_for_status()

    data = r.json()
    path = []
    for sec in data["routes"][0]["sections"]:
        for road in sec["roads"]:
            for i in range(0, len(road["vertexes"]), 2):
                lng = road["vertexes"][i]
                lat = road["vertexes"][i + 1]
                path.append((lat, lng))

    return path


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def min_distance_to_route(st_lat, st_lng, route_points):
    dists = [haversine(st_lat, st_lng, lat, lng) for lat, lng in route_points]
    return min(dists) if dists else 999999