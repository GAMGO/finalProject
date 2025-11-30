# app/pipelines/recommend_pipeline.py
import pandas as pd
from app.repositories.stall_repository import get_all_stalls
from app.services.route_service import get_route
from app.services.stall_service import filter_stalls_by_radius
from app.services.recommend_service import weighted_recommend
from app.models.ml.stall_recommender import StallRecommender

def recommend_pipeline(start, dest, via, user_id):

    # 1) Load stalls
    rows = get_all_stalls()
    stalls = pd.DataFrame(rows, columns=[
        "idx","name","lat","lng","stall_category_id",
        "rating_avg","rating_count_log",
        "sentiment_score","price_level","hour_sin",
        "user_id"
    ])

    # 2) Get route points
    route_points = get_route(start, dest, via)

    # 3) Filter by radii
    near50 = filter_stalls_by_radius(stalls, route_points, 50)
    near100 = filter_stalls_by_radius(stalls, route_points, 100)
    near200 = filter_stalls_by_radius(stalls, route_points, 200)

    candidates = pd.concat([near50, near100, near200]).drop_duplicates()

    # 4) Prepare ML model
    user2idx = {u: i for i, u in enumerate(stalls["user_id"].unique())}
    cat2idx = {c: i for i, c in enumerate(stalls["stall_category_id"].unique())}

    model = StallRecommender(
        num_users=len(user2idx),
        num_categories=len(cat2idx),
        numeric_dim=6
    )
    model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
    model.eval()

    # 5) Weighted recommendation
    result = weighted_recommend(
        df=candidates,
        model=model,
        user_id=user_id,
        user2idx=user2idx,
        cat2idx=cat2idx,
        top_k=3
    )

    return result.to_dict(orient="records")