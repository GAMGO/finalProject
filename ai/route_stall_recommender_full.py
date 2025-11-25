from dotenv import load_dotenv
import os
import math
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

load_dotenv()

# ============================================================
# 0. 환경변수에서 Kakao key 로드
# ============================================================
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"


# ============================================================
# 1. Kakao 경로 API
# ============================================================
def get_route_from_kakao(start, destination, waypoints=None, priority="TIME"):
    if not KAKAO_REST_API_KEY:
        raise RuntimeError("환경변수 KAKAO_REST_API_KEY가 없습니다.")

    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}

    if waypoints:
        waypoints_param = "|".join(
            f"{wp['lng']},{wp['lat']}" for wp in waypoints
        )
    else:
        waypoints_param = ""

    params = {
        "origin": f"{start['lng']},{start['lat']}",
        "destination": f"{destination['lng']},{destination['lat']}",
        "priority": priority,
    }

    if waypoints_param:
        params["waypoints"] = waypoints_param

    r = requests.get(DIRECTIONS_URL, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    routes = data.get("routes", [])
    if not routes:
        raise RuntimeError("routes 없음.")

    route = routes[0]
    sections = route["sections"]

    path = []
    for sec in sections:
        for road in sec["roads"]:
            v = road["vertexes"]
            for i in range(0, len(v), 2):
                lng = v[i]
                lat = v[i + 1]
                path.append((lat, lng))

    return {
        "summary": route["summary"],
        "path": path
    }


# ============================================================
# 2. 거리 계산 유틸
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def min_distance_to_route(st_lat, st_lng, route_points):
    if not route_points:
        return 999999.0
    return min(
        haversine(st_lat, st_lng, lat, lng)
        for (lat, lng) in route_points
    )


# ============================================================
# 3. 반경별 노점 필터링
# ============================================================
def filter_stalls_by_radius(stalls_df, route_points, radius_m):
    dists = []
    for _, r in stalls_df.iterrows():
        d = min_distance_to_route(r["lat"], r["lng"], route_points)
        dists.append(d)

    stalls_df = stalls_df.copy()
    stalls_df["distance_to_route_m"] = dists
    return stalls_df[stalls_df["distance_to_route_m"] <= radius_m].sort_values("distance_to_route_m")


# ============================================================
# 4. 더미 노점 생성 (DB 대체)
# ============================================================
def build_dummy_stalls(center_lat, center_lng, n=80):
    np.random.seed(0)
    lats = center_lat + (np.random.rand(n) - 0.5) * 0.015
    lngs = center_lng + (np.random.rand(n) - 0.5) * 0.015

    data = []
    for i in range(n):
        data.append({
            "stall_id": i + 1,
            "name": f"노점_{i+1}",
            "lat": lats[i],
            "lng": lngs[i],
            "stall_category_id": np.random.randint(1, 5),
            "rating_avg": np.random.uniform(2, 5),
            "rating_count_log": np.log1p(np.random.randint(1, 200)),
            "sentiment_score": np.random.uniform(-1, 1),
            "price_level": np.random.randint(1, 4),
        })
    return pd.DataFrame(data)


# ============================================================
# 5. PyTorch 추천 모델
# ============================================================
class StallModel(nn.Module):
    def __init__(self, num_users, num_categories):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 16)
        self.cat_emb = nn.Embedding(num_categories, 8)

        self.numeric_dim = 6  # distance_from_route + rating + etc.

        self.mlp = nn.Sequential(
            nn.Linear(16 + 8 + self.numeric_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, user_id, cat_id, numeric):
        x = torch.cat([self.user_emb(user_id), self.cat_emb(cat_id), numeric], dim=1)
        return self.mlp(x).squeeze(-1)


class StallDataset(Dataset):
    def __init__(self, df, user2idx, cat2idx):
        self.df = df.reset_index(drop=True)
        self.user2idx = user2idx
        self.cat2idx = cat2idx
        self.num_cols = [
            "distance_from_route",
            "rating_avg",
            "rating_count_log",
            "sentiment_score",
            "price_level",
            "hour_sin",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        # user / category
        user_raw = r["user_id"]
        cat_raw = r["stall_category_id"]
        user = torch.tensor(self.user2idx[user_raw], dtype=torch.long)
        cat = torch.tensor(self.cat2idx[cat_raw], dtype=torch.long)

        # 🔥 numeric: object / str 방지용 강제 float 변환
        numeric_vals = []
        for col in self.num_cols:
            v = r[col]
            try:
                v = float(v)
            except Exception:
                v = 0.0
            numeric_vals.append(v)
        numeric = torch.tensor(numeric_vals, dtype=torch.float32)

        # label
        try:
            label_val = float(r["label"])
        except Exception:
            label_val = 0.0
        label = torch.tensor(label_val, dtype=torch.float32)

        return user, cat, numeric, label


def train_model(model, train_loader, val_loader, device="cpu"):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(3):
        model.train()
        tot = 0.0
        for u, c, n, y in train_loader:
            u, c, n, y = u.to(device), c.to(device), n.to(device), y.to(device)
            opt.zero_grad()
            pred = model(u, c, n)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tot += loss.item()

        print(f"[Epoch {epoch + 1}] TrainLoss={tot:.4f}")

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for u, c, n, y in val_loader:
                u, c, n, y = u.to(device), c.to(device), n.to(device), y.to(device)
                loss = loss_fn(model(u, c, n), y)
                vtot += loss.item()
        print(f"           ValLoss={vtot:.4f}")


# ============================================================
# 6. 가중치 추첨 추천
# ============================================================
@torch.no_grad()
def recommend(model, df, user_id, user2idx, cat2idx, k=3, temperature=0.7, device="cpu"):
    if len(df) == 0:
        print("⚠️ 추천 후보가 없습니다.")
        return df

    df = df.copy()

    numeric_cols = [
        "distance_from_route",
        "rating_avg",
        "rating_count_log",
        "sentiment_score",
        "price_level",
        "hour_sin",
    ]

    # numeric 안전 변환
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    user_idx = torch.tensor([user2idx[user_id]] * len(df), dtype=torch.long, device=device)
    cat_idx = torch.tensor([cat2idx[c] for c in df["stall_category_id"]], dtype=torch.long, device=device)
    numeric = torch.tensor(df[numeric_cols].values, dtype=torch.float32, device=device)

    scores = model(user_idx, cat_idx, numeric)
    probs = F.softmax(scores / temperature, dim=0)

    k = min(k, len(df))
    chosen = torch.multinomial(probs, k, replacement=False).cpu().numpy()

    df["score"] = scores.cpu().numpy()
    df["prob"] = probs.cpu().numpy()

    return df.iloc[chosen]


# ============================================================
# 7. 실행
# ============================================================
if __name__ == "__main__":
    # 1) 경로 설정
    start = {"lat": 37.5665, "lng": 126.9780}      # 서울 시청
    destination = {"lat": 37.4979, "lng": 127.0276}  # 강남역
    waypoints = [{"lat": 37.5048, "lng": 127.0041}]  # 고속터미널 근처

    print("🔹 Kakao 경로 요청...")
    route = get_route_from_kakao(start, destination, waypoints)
    path = route["path"]
    mid_lat, mid_lng = path[len(path) // 2]

    print("🚚 경로 좌표 수:", len(path))

    # 2) 노점 생성
    stalls = build_dummy_stalls(mid_lat, mid_lng, n=100)

    # 3) 반경별 필터링
    print("\n📍 반경별 후보 노점 수")
    near50 = filter_stalls_by_radius(stalls, path, 50)
    near100 = filter_stalls_by_radius(stalls, path, 100)
    near200 = filter_stalls_by_radius(stalls, path, 200)

    print("50m :", len(near50))
    print("100m:", len(near100))
    print("200m:", len(near200))

    # ---------------------------------------------
    # 4) PyTorch 모델 학습용 데이터 준비
    # ---------------------------------------------
    stalls["user_id"] = np.random.randint(1, 20, size=len(stalls))
    stalls["hour_sin"] = np.sin(np.random.randint(0, 24, size=len(stalls)) / 3.14)
    stalls["label"] = np.where(stalls["rating_avg"] > 4.5, 1, 0).astype(float)

    # distance_from_route: 학습용 더미 값
    stalls["distance_from_route"] = np.random.uniform(10, 400, size=len(stalls))

    # numeric 컬럼 float 강제
    numeric_cols_fix = [
        "distance_from_route",
        "rating_avg",
        "rating_count_log",
        "sentiment_score",
        "price_level",
        "hour_sin",
    ]
    for col in numeric_cols_fix:
        stalls[col] = pd.to_numeric(stalls[col], errors="coerce").fillna(0.0).astype(float)

    stalls = stalls.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    user2idx = {u: i for i, u in enumerate(stalls["user_id"].unique())}
    cat2idx = {c: i for i, c in enumerate(stalls["stall_category_id"].unique())}

    train = stalls.sample(frac=0.8, random_state=0)
    val = stalls.drop(train.index)

    train_loader = DataLoader(StallDataset(train, user2idx, cat2idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(StallDataset(val, user2idx, cat2idx), batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StallModel(len(user2idx), len(cat2idx)).to(device)

    print("\n🧠 PyTorch 모델 학습")
    train_model(model, train_loader, val_loader, device=device)

    TEST_USER = list(user2idx.keys())[0]

    # ---------------------------------------------
    # 5) 반경별 추천 (distance_from_route를 실제 거리로 설정)
    # ---------------------------------------------
    print("\n🎯 50m 추천:")
    if len(near50) > 0:
        near50 = near50.copy()
        near50["user_id"] = TEST_USER
        near50["hour_sin"] = np.sin(np.random.randint(0, 24, size=len(near50)) / 3.14)
        near50["distance_from_route"] = near50["distance_to_route_m"]
        print(recommend(model, near50, TEST_USER, user2idx, cat2idx, k=3, device=device))

    print("\n🎯 100m 추천:")
    if len(near100) > 0:
        near100 = near100.copy()
        near100["user_id"] = TEST_USER
        near100["hour_sin"] = np.sin(np.random.randint(0, 24, size=len(near100)) / 3.14)
        near100["distance_from_route"] = near100["distance_to_route_m"]
        print(recommend(model, near100, TEST_USER, user2idx, cat2idx, k=3, device=device))

    print("\n🎯 200m 추천:")
    if len(near200) > 0:
        near200 = near200.copy()
        near200["user_id"] = TEST_USER
        near200["hour_sin"] = np.sin(np.random.randint(0, 24, size=len(near200)) / 3.14)
        near200["distance_from_route"] = near200["distance_to_route_m"]
        print(recommend(model, near200, TEST_USER, user2idx, cat2idx, k=3, device=device))