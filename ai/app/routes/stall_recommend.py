# app/routes/stall_recommend.py
from fastapi import APIRouter
from app.services.stall_recommendation import recommend_with_review_summary

router = APIRouter(prefix="/api/recommend")

@router.post("/stalls")
def recommend_stalls(payload: dict):
    start = payload.get("start")
    waypoint = payload.get("waypoint")
    end = payload.get("end")
    radius_list = payload.get("radius_list", [50, 100, 200])

    def process_point(coord):
        if not coord:
            return None

        lat = coord["lat"]
        lng = coord["lng"]

        data = {}
        for r in radius_list:
            data[f"{r}m"] = recommend_with_review_summary(lat, lng, r)
        return data

    return {
        "start": process_point(start),
        "waypoint": process_point(waypoint),
        "end": process_point(end)
    }