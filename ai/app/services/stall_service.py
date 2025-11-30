# app/services/stall_service.py
import pandas as pd
from app.services.route_service import min_distance_to_route

def filter_stalls_by_radius(stalls_df, route_points, radius_m):
    result = []
    for _, r in stalls_df.iterrows():
        d = min_distance_to_route(r["lat"], r["lng"], route_points)
        if d <= radius_m:
            row = r.copy()
            row["distance_from_route"] = d
            result.append(row)
    return pd.DataFrame(result).sort_values("distance_from_route")