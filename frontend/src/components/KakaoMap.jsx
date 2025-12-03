// src/components/KakaoMap.jsx
import React, { useEffect, useRef, useState } from "react";
import plusIcon from "../assets/plus.svg";
import "./KakaoMap.css";

const APP_KEY = "bdd84bdbed2db3bc5d8b90cd6736a995";
const API_BASE = "http://localhost:8080";

// FOOD_INFO / FoodCategory 기준 (백엔드에 아직 안 쓰여도 프론트용)
const CATEGORIES = [
  { id: 1, label: "통닭" },
  { id: 2, label: "타코야끼" },
  { id: 3, label: "순대곱창" },
  { id: 4, label: "붕어빵" },
  { id: 5, label: "군밤/고구마" },
  { id: 6, label: "닭꼬치" },
  { id: 7, label: "분식" },
  { id: 8, label: "해산물" },
  { id: 9, label: "뻥튀기" },
  { id: 10, label: "계란빵" },
  { id: 11, label: "옥수수" },
];

// ✅ Store 객체에서 PK 꺼내는 공통 헬퍼 (idx / id / storeIdx 아무거나 올 수 있음)
const getStoreIdx = (store) => {
  if (!store) return null;
  return store.idx ?? store.id ?? store.storeIdx ?? store.store_id ?? null;
};

export default function KakaoMap() {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const geocoderRef = useRef(null);
  const tempMarkerRef = useRef(null);
  const markersRef = useRef([]);

  // ✅ 길찾기용
  const routeLineRef = useRef(null);
  const placesRef = useRef(null);

  // ===== 노점 등록 모달 =====
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPos, setSelectedPos] = useState(null);
  const [form, setForm] = useState({
    categoryId: "",
    address: "",
    description: "",
  });
  const [isPickingLocation, setIsPickingLocation] = useState(false);
  const isPickingLocationRef = useRef(false);

  // ===== 상세 + 리뷰 모달 =====
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [selectedStore, setSelectedStore] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [reviewStats, setReviewStats] = useState(null);
  const [reviewsLoading, setReviewsLoading] = useState(false);
  const [reviewSubmitting, setReviewSubmitting] = useState(false);
  const [reviewForm, setReviewForm] = useState({
    rating: 5,
    text: "",
  });
  const [hoverRating, setHoverRating] = useState(0);

  // ===== 길찾기 상태 =====
  const [routeForm, setRouteForm] = useState({ from: "", to: "" });
  const [routeMode, setRouteMode] = useState("CAR"); // CAR / WALK / TRANSIT
  const [routeLoading, setRouteLoading] = useState(false);
  const [routeError, setRouteError] = useState("");

  // ==========================
  // 유틸
  // ==========================
  const formatDateTime = (str) => {
    if (!str) return "";
    return str.replace("T", " ").slice(0, 16);
  };

  const getAvgRatingText = () => {
    if (!reviewStats || reviewStats.avgRating == null) return "0.0";
    const n =
      typeof reviewStats.avgRating === "number"
        ? reviewStats.avgRating
        : Number(reviewStats.avgRating);
    if (Number.isNaN(n)) return "0.0";
    return n.toFixed(1);
  };

  const renderStars = (value) => {
    const num = typeof value === "number" ? value : Number(value || 0);
    const rounded = Math.round(num);

    return (
      <span style={{ fontSize: 18, color: "#facc15" }}>
        {[1, 2, 3, 4, 5].map((i) => (
          <span key={i}>{i <= rounded ? "★" : "☆"}</span>
        ))}
      </span>
    );
  };

  // ==========================
  // 리뷰 + 통계 불러오기 (/with-stats 사용)
  // ==========================
  const loadReviews = async (storeIdx) => {
    if (!storeIdx) {
      console.warn("loadReviews: storeIdx가 없습니다.", storeIdx);
      return;
    }

    setReviewsLoading(true);
    try {
      const res = await fetch(
        `${API_BASE}/api/stores/${storeIdx}/reviews/with-stats?page=0&size=20`
      );
      const text = await res.text();
      console.log(
        "GET /api/stores/{id}/reviews/with-stats:",
        res.status,
        text
      );

      if (!res.ok) {
        console.error("리뷰+통계 불러오기 실패:", res.status, text);
        setReviews([]);
        setReviewStats(null);
        return;
      }

      const json = JSON.parse(text);
      const data = json.data ?? json; // ApiResponse 래퍼 고려

      setReviewStats(data.stats || null);
      setReviews(Array.isArray(data.reviews) ? data.reviews : []);
    } catch (err) {
      console.error("리뷰+통계 불러오기 에러:", err);
      setReviews([]);
      setReviewStats(null);
    } finally {
      setReviewsLoading(false);
    }
  };

  const handleMarkerClick = (store) => {
    const storeIdx = getStoreIdx(store);
    console.log("marker click store:", store, "idx:", storeIdx);
    setSelectedStore(store);
    setIsDetailOpen(true);
    setReviewForm({ rating: 5, text: "" });
    setHoverRating(0);
    loadReviews(storeIdx);
  };

  const closeDetail = () => {
    setIsDetailOpen(false);
    setSelectedStore(null);
    setReviews([]);
    setReviewStats(null);
    setReviewForm({ rating: 5, text: "" });
    setHoverRating(0);
  };

  // ==========================
  // 가게 / 마커
  // ==========================
  const addStoreMarker = (map, store) => {
    if (!window.kakao || !map || !store) return;

    // ✅ DTO(StoreResponse) latitude/longitude + 엔티티 lat/lng 둘 다 지원
    const lat = store.latitude ?? store.lat;
    const lng = store.longitude ?? store.lng;

    if (lat == null || lng == null) {
      console.warn("마커 좌표 없음, store:", store);
      return;
    }

    const position = new window.kakao.maps.LatLng(lat, lng);

    const marker = new window.kakao.maps.Marker({
      position,
      map,
    });

    const categoryText = store.category ?? "";
    const nameText = store.storeName ?? store.name ?? "";
    const addressText = store.address ?? store.storeAddress ?? "";

    const content = `
      <div style="padding:8px 12px;font-size:12px;max-width:220px;">
        ${
          categoryText
            ? `<div style="font-weight:600;margin-bottom:4px;">${categoryText}</div>`
            : ""
        }
        ${nameText ? `<div style="margin-bottom:4px;">${nameText}</div>` : ""}
        ${
          addressText
            ? `<div style="font-size:11px;color:#555;">${addressText}</div>`
            : ""
        }
      </div>
    `;

    const infowindow = new window.kakao.maps.InfoWindow({ content });

    window.kakao.maps.event.addListener(marker, "click", () => {
      infowindow.open(map, marker);
      handleMarkerClick(store);
    });

    markersRef.current.push({ marker, infowindow });
  };

  const loadStoresAndDraw = async (map) => {
    try {
      const res = await fetch(`${API_BASE}/api/stores`);
      const text = await res.text();
      console.log("GET /api/stores:", res.status, text);

      if (!res.ok) {
        console.error("가게 목록 불러오기 실패:", res.status, text);
        return;
      }

      let json;
      try {
        json = JSON.parse(text);
      } catch (e) {
        console.error("가게 목록 JSON 파싱 실패:", e);
        return;
      }

      // ✅ 배열 그대로 오거나, { data: [...] } 래핑된 경우 둘 다 처리
      const stores = Array.isArray(json) ? json : json.data || [];
      stores.forEach((s) => addStoreMarker(map, s));
    } catch (err) {
      console.error("가게 목록 불러오기 실패:", err);
    }
  };

  // ==========================
  // 지도 초기화
  // ==========================
  useEffect(() => {
    const scriptId = "kakao-map-sdk";

    const initMap = () => {
      if (!window.kakao || !window.kakao.maps) {
        console.log("[KAKAO] kakao.maps not ready");
        return;
      }
      if (!mapRef.current) {
        console.log("[KAKAO] mapRef is null");
        return;
      }

      window.kakao.maps.load(async () => {
        const center = new window.kakao.maps.LatLng(37.5665, 126.978);
        const options = { center, level: 4 };

        const map = new window.kakao.maps.Map(mapRef.current, options);
        mapInstanceRef.current = map;

        geocoderRef.current = new window.kakao.maps.services.Geocoder();

        // ✅ 장소 검색 객체 생성 (길찾기에서 사용)
        placesRef.current = new window.kakao.maps.services.Places();

        window.kakao.maps.event.addListener(map, "click", (mouseEvent) => {
          const latlng = mouseEvent.latLng;
          const lat = latlng.getLat();
          const lng = latlng.getLng();

          setSelectedPos({ lat, lng });

          if (!tempMarkerRef.current) {
            tempMarkerRef.current = new window.kakao.maps.Marker({
              position: latlng,
              map,
            });
          } else {
            tempMarkerRef.current.setPosition(latlng);
          }

          if (geocoderRef.current) {
            geocoderRef.current.coord2Address(
              lng,
              lat,
              (result, status) => {
                if (status === window.kakao.maps.services.Status.OK) {
                  const addr =
                    result[0].road_address?.address_name ||
                    result[0].address.address_name;
                  setForm((prev) => ({ ...prev, address: addr || "" }));
                }
              }
            );
          }

          if (isPickingLocationRef.current) {
            setIsModalOpen(true);
            setIsPickingLocation(false);
            isPickingLocationRef.current = false;
          }
        });

        console.log("[KAKAO] map created", map);
        await loadStoresAndDraw(map);
      });
    };

    const existing = document.getElementById(scriptId);

    if (!existing) {
      const script = document.createElement("script");
      script.id = scriptId;
      script.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${APP_KEY}&autoload=false&libraries=services`;
      script.async = true;
      script.onload = () => {
        console.log("[KAKAO] script loaded");
        initMap();
      };
      script.onerror = (e) => {
        console.error("[KAKAO] script load error", e);
      };
      document.head.appendChild(script);
    } else {
      console.log("[KAKAO] script loaded (from existing)");
      initMap();
    }
  }, []);

  // ==========================
  // 노점 등록 모달
  // ==========================
  const openModal = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setIsPickingLocation(false);
    isPickingLocationRef.current = false;
    setForm({ categoryId: "", address: "", description: "" });
    setSelectedPos(null);
    if (tempMarkerRef.current) {
      tempMarkerRef.current.setMap(null);
      tempMarkerRef.current = null;
    }
  };

  const handleStartPickLocation = () => {
    setIsPickingLocation(true);
    isPickingLocationRef.current = true;
    setIsModalOpen(false);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedPos) {
      alert("지도를 클릭해서 위치를 먼저 선택해줘요");
      return;
    }

    const payload = {
      storeName: form.description || "이름 없는 노점",
      storeAddress: form.address || "",
      lat: selectedPos.lat,
      lng: selectedPos.lng,
    };

    try {
      const res = await fetch(`${API_BASE}/api/stores`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const text = await res.text();
      console.log("POST /api/stores:", res.status, text);

      if (!res.ok) {
        alert(`가게 등록 실패 (${res.status})`);
        return;
      }

      let savedId = null;
      try {
        const json = JSON.parse(text);
        if (typeof json === "number") {
          savedId = json;
        } else if (json && typeof json === "object") {
          if (typeof json.data === "number") savedId = json.data;
          else if (typeof json.id === "number") savedId = json.id;
        }
      } catch {
        const n = Number(text);
        if (!Number.isNaN(n)) savedId = n;
      }

      if (mapInstanceRef.current) {
        const newStoreForMarker = {
          idx: savedId,
          storeName: payload.storeName,
          storeAddress: payload.storeAddress,
          lat: payload.lat,
          lng: payload.lng,
        };
        addStoreMarker(mapInstanceRef.current, newStoreForMarker);
      }

      closeModal();
    } catch (err) {
      console.error("가게 등록 실패:", err);
      alert("가게 등록에 실패했어 ㅠㅠ 콘솔 로그 한 번 봐줘.");
    }
  };

  // ==========================
  // 리뷰 작성
  // ==========================
  const handleReviewFormChange = (e) => {
    const { name, value } = e.target;
    setReviewForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleReviewSubmit = async (e) => {
    e.preventDefault();
    if (!selectedStore) {
      alert("선택된 노점이 없습니다.");
      return;
    }

    const storeIdx = getStoreIdx(selectedStore);
    if (!storeIdx) {
      alert("가게 ID를 찾을 수 없어서 리뷰를 저장할 수 없습니다.");
      console.error("handleReviewSubmit: storeIdx 없음", selectedStore);
      return;
    }

    // 토큰 키 여러 개 중 하나라도 있으면 사용
    const token =
      localStorage.getItem("jwtToken") ||
      localStorage.getItem("accessToken") ||
      localStorage.getItem("token");

    if (!token) {
      alert("로그인 후 리뷰를 작성할 수 있어요.");
      return;
    }

    const ratingNum = Number(reviewForm.rating);
    if (!ratingNum || ratingNum < 1 || ratingNum > 5) {
      alert("평점은 1~5 사이 숫자만 가능합니다.");
      return;
    }

    const payload = {
      rating: ratingNum,
      reviewText: reviewForm.text || "",
    };

    setReviewSubmitting(true);
    try {
      const res = await fetch(
        `${API_BASE}/api/stores/${storeIdx}/reviews`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(payload),
        }
      );

      const text = await res.text();
      console.log("POST /api/stores/{id}/reviews:", res.status, text);

      if (!res.ok) {
        console.error("리뷰 작성 실패:", res.status, text);
        alert(
          `리뷰 등록에 실패했어 ㅠㅠ\n(status: ${res.status})\n콘솔 로그도 한 번 봐줘.`
        );
        return;
      }

      setReviewForm({ rating: 5, text: "" });
      setHoverRating(0);
      await loadReviews(storeIdx);
    } catch (err) {
      console.error("리뷰 작성 에러:", err);
      alert("리뷰 등록 중 에러가 발생했어 ㅠㅠ");
    } finally {
      setReviewSubmitting(false);
    }
  };

  // ==========================
  // 길찾기 (출발/도착 입력 → 경로 그리기)
  // ==========================
  const handleRouteChange = (e) => {
    const { name, value } = e.target;
    setRouteForm((prev) => ({ ...prev, [name]: value }));
  };

  const clearRoute = () => {
    setRouteForm({ from: "", to: "" });
    setRouteError("");
    setRouteLoading(false);
    setRouteMode("CAR");
    if (routeLineRef.current) {
      routeLineRef.current.setMap(null);
      routeLineRef.current = null;
    }
  };

  const handleRouteSearch = async (e) => {
    e.preventDefault();
    if (!mapInstanceRef.current || !window.kakao) return;

    const { from, to } = routeForm;
    if (!from || !to) {
      setRouteError("출발지와 도착지를 모두 입력해 주세요.");
      return;
    }

    const places = placesRef.current;
    if (!places) {
      setRouteError("카카오 장소 검색을 초기화하지 못했습니다.");
      return;
    }

    const searchKeyword = (keyword) =>
      new Promise((resolve, reject) => {
        places.keywordSearch(keyword, (data, status) => {
          if (
            status === window.kakao.maps.services.Status.OK &&
            data &&
            data.length > 0
          ) {
            resolve(data[0]);
          } else {
            reject(new Error(`주소 변환 실패: ${keyword}`));
          }
        });
      });

    try {
      setRouteLoading(true);
      setRouteError("");

      const [fromPlace, toPlace] = await Promise.all([
        searchKeyword(from),
        searchKeyword(to),
      ]);

      const fromPoint = {
        lat: parseFloat(fromPlace.y),
        lng: parseFloat(fromPlace.x),
      };
      const toPoint = {
        lat: parseFloat(toPlace.y),
        lng: parseFloat(toPlace.x),
      };

      const res = await fetch(`${API_BASE}/api/routes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          from: fromPoint,
          to: toPoint,
          mode: routeMode, // 🔥 이동수단 같이 전송
        }),
      });

      const text = await res.text();
      console.log("POST /api/routes:", res.status, text);

      if (!res.ok) {
        throw new Error(`길찾기 실패 (${res.status})`);
      }

      const json = JSON.parse(text);
      const data = json.data ?? json;

      const points = Array.isArray(data?.path)
        ? data.path
        : Array.isArray(data?.points)
        ? data.points
        : [];

      if (!points.length) {
        throw new Error("경로 데이터가 비어 있습니다.");
      }

      if (routeLineRef.current) {
        routeLineRef.current.setMap(null);
        routeLineRef.current = null;
      }

      const path = points.map(
        (p) => new window.kakao.maps.LatLng(p.lat, p.lng)
      );

      const strokeColor =
        routeMode === "WALK"
          ? "#16a34a"
          : routeMode === "TRANSIT"
          ? "#a855f7"
          : "#2563eb";

      const polyline = new window.kakao.maps.Polyline({
        path,
        strokeWeight: 5,
        strokeColor,
        strokeOpacity: 0.9,
        strokeStyle: "solid",
      });
      polyline.setMap(mapInstanceRef.current);
      routeLineRef.current = polyline;

      const bounds = new window.kakao.maps.LatLngBounds();
      path.forEach((latlng) => bounds.extend(latlng));
      mapInstanceRef.current.setBounds(bounds);
    } catch (err) {
      console.error("길찾기 에러:", err);
      setRouteError(err.message || "길찾기 중 에러가 발생했습니다.");
    } finally {
      setRouteLoading(false);
    }
  };

  // ==========================
  // 렌더
  // ==========================
  return (
    <>
      {/* 지도 */}
      <div
        style={{
          position: "relative",
          width: "100vw",
          height: "100vh",
        }}
      >
        <div
          ref={mapRef}
          style={{
            width: "100%",
            height: "100%",
          }}
        />
      </div>

      {/* 오른쪽 위 길찾기 패널 */}
      <div
        style={{
          position: "fixed",
          top: "16px",
          right: "24px",
          zIndex: 10000,
          background: "rgba(255,255,255,0.96)",
          borderRadius: 12,
          boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
          padding: "10px 12px",
          width: 280,
          fontSize: 12,
        }}
      >
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            marginBottom: 8,
          }}
        >
          길찾기
        </div>

        {/* 이동 수단 선택 */}
        <div
          style={{
            display: "flex",
            gap: 8,
            marginBottom: 6,
            fontSize: 12,
            alignItems: "center",
          }}
        >
          <span style={{ color: "#6b7280", marginRight: 4 }}>이동 수단</span>
          <label style={{ display: "flex", alignItems: "center", gap: 2 }}>
            <input
              type="radio"
              name="routeMode"
              value="CAR"
              checked={routeMode === "CAR"}
              onChange={(e) => setRouteMode(e.target.value)}
            />
            차량
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 2 }}>
            <input
              type="radio"
              name="routeMode"
              value="WALK"
              checked={routeMode === "WALK"}
              onChange={(e) => setRouteMode(e.target.value)}
            />
            도보
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 2 }}>
            <input
              type="radio"
              name="routeMode"
              value="TRANSIT"
              checked={routeMode === "TRANSIT"}
              onChange={(e) => setRouteMode(e.target.value)}
            />
            대중교통
          </label>
        </div>

        <form onSubmit={handleRouteSearch}>
          <div style={{ marginBottom: 6 }}>
            <div style={{ marginBottom: 2 }}>출발</div>
            <input
              name="from"
              value={routeForm.from}
              onChange={handleRouteChange}
              placeholder="예: 서울역"
              style={{
                width: "100%",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                padding: "4px 8px",
              }}
            />
          </div>
          <div style={{ marginBottom: 6 }}>
            <div style={{ marginBottom: 2 }}>도착</div>
            <input
              name="to"
              value={routeForm.to}
              onChange={handleRouteChange}
              placeholder="예: 시청역"
              style={{
                width: "100%",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                padding: "4px 8px",
              }}
            />
          </div>
          {routeError && (
            <div
              style={{
                color: "#dc2626",
                fontSize: 11,
                marginBottom: 4,
                whiteSpace: "pre-wrap",
              }}
            >
              {routeError}
            </div>
          )}
          <div
            style={{
              display: "flex",
              justifyContent: "flex-end",
              gap: 6,
              marginTop: 4,
            }}
          >
            <button
              type="button"
              onClick={clearRoute}
              style={{
                borderRadius: 999,
                border: "1px solid #e5e7eb",
                background: "#fff",
                padding: "4px 10px",
                cursor: "pointer",
              }}
            >
              초기화
            </button>
            <button
              type="submit"
              disabled={routeLoading}
              style={{
                borderRadius: 999,
                border: "none",
                background: routeLoading ? "#9ca3af" : "#2563eb",
                color: "#fff",
                padding: "4px 10px",
                fontWeight: 600,
                cursor: routeLoading ? "default" : "pointer",
              }}
            >
              {routeLoading ? "검색 중..." : "길찾기"}
            </button>
          </div>
        </form>
      </div>

      {/* 오른쪽 아래 플로팅 버튼 */}
      <button
        type="button"
        style={{
          position: "fixed",
          right: "24px",
          bottom: "24px",
          width: "56px",
          height: "56px",
          border: "none",
          padding: 0,
          background: "transparent",
          cursor: "pointer",
          zIndex: 9999,
        }}
        onClick={openModal}
      >
        <img
          src={plusIcon}
          alt="노점 추가"
          style={{
            width: "100%",
            height: "100%",
            display: "block",
          }}
        />
      </button>

      {/* 노점 등록 모달 */}
      {isModalOpen && (
        <div className="map-modal-backdrop" onClick={closeModal}>
          <div className="map-modal" onClick={(e) => e.stopPropagation()}>
            <h3 className="map-modal-title">노점 추가</h3>

            <form onSubmit={handleSubmit}>
              <label className="map-label">카테고리</label>
              <select
                name="categoryId"
                value={form.categoryId}
                onChange={handleChange}
                className="map-select"
              >
                <option value="">선택해주세요</option>
                {CATEGORIES.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </select>

              <label className="map-label">주소 (직접 수정 가능)</label>
              <input
                type="text"
                name="address"
                value={form.address}
                onChange={handleChange}
                placeholder="지도를 클릭하면 자동으로 채워져요"
                className="map-input"
              />

              <label className="map-label">노점 설명</label>
              <textarea
                name="description"
                value={form.description}
                onChange={handleChange}
                rows={4}
                placeholder="예: 매일 저녁 7시~11시, 순살통닭/감자튀김 판매 등"
                className="map-textarea"
              />

              <div className="map-pick-row">
                <button
                  type="button"
                  onClick={handleStartPickLocation}
                  className="map-pick-button"
                >
                  지도에서 위치 선택하기
                </button>
              </div>

              <div className="map-modal-actions">
                <button
                  type="button"
                  onClick={closeModal}
                  className="map-btn-cancel"
                >
                  취소
                </button>
                <button type="submit" className="map-btn-submit">
                  등록하기
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* 상세 + 리뷰 모달 */}
      {isDetailOpen && selectedStore && (
        <div className="map-modal-backdrop" onClick={closeDetail}>
          <div
            className="map-modal"
            onClick={(e) => e.stopPropagation()}
            style={{ maxWidth: 520 }}
          >
            {/* 헤더 */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 8,
              }}
            >
              <h3 className="map-modal-title">
                {selectedStore.category && (
                  <span
                    style={{
                      fontSize: 13,
                      padding: "2px 8px",
                      borderRadius: 999,
                      background: "#f3f4f6",
                      marginRight: 8,
                    }}
                  >
                    {selectedStore.category}
                  </span>
                )}
                {selectedStore.storeName || "이름 없는 노점"}
              </h3>
              <button
                type="button"
                onClick={closeDetail}
                style={{
                  border: "none",
                  background: "transparent",
                  fontSize: 18,
                  cursor: "pointer",
                }}
              >
                ✕
              </button>
            </div>

            {/* 주소 */}
            {selectedStore.address || selectedStore.storeAddress ? (
              <div
                style={{
                  fontSize: 13,
                  color: "#4b5563",
                  marginBottom: 12,
                }}
              >
                📍 {selectedStore.address || selectedStore.storeAddress}
              </div>
            ) : null}

            {/* 평균 별점 */}
            <div
              style={{
                padding: "10px 12px",
                borderRadius: 8,
                background: "#f9fafb",
                marginBottom: 14,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <div>
                <div style={{ fontSize: 13, color: "#6b7280" }}>
                  평균 별점
                </div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  {renderStars(reviewStats?.avgRating)}
                  <span style={{ fontWeight: 600, fontSize: 16 }}>
                    {getAvgRatingText()}
                  </span>
                  <span style={{ fontSize: 12, color: "#6b7280" }}>
                    ({reviewStats?.ratingCount || 0}개)
                  </span>
                </div>
              </div>
            </div>

            {/* 리뷰 작성 */}
            <form onSubmit={handleReviewSubmit} style={{ marginBottom: 16 }}>
              {/* 별점 선택 (별 클릭) */}
              <div style={{ marginBottom: 8 }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                  }}
                >
                  <label
                    style={{ fontSize: 13, fontWeight: 600 }}
                  >
                    별점
                  </label>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                    }}
                  >
                    {[1, 2, 3, 4, 5].map((star) => {
                      const current = hoverRating || reviewForm.rating;
                      const filled = star <= current;
                      return (
                        <button
                          key={star}
                          type="button"
                          onClick={() =>
                            setReviewForm((prev) => ({
                              ...prev,
                              rating: star,
                            }))
                          }
                          onMouseEnter={() => setHoverRating(star)}
                          onMouseLeave={() => setHoverRating(0)}
                          style={{
                            border: "none",
                            background: "transparent",
                            padding: 0,
                            cursor: "pointer",
                            fontSize: 24,
                            lineHeight: 1,
                            color: "#facc15",
                          }}
                        >
                          {filled ? "★" : "☆"}
                        </button>
                      );
                    })}
                    <span
                      style={{
                        fontSize: 13,
                        color: "#374151",
                        marginLeft: 4,
                      }}
                    >
                      {reviewForm.rating}점
                    </span>
                  </div>
                </div>
              </div>

              <textarea
                name="text"
                value={reviewForm.text}
                onChange={handleReviewFormChange}
                rows={3}
                placeholder="노점에 대한 리뷰를 남겨주세요."
                style={{
                  width: "100%",
                  resize: "vertical",
                  padding: "8px 10px",
                  borderRadius: 8,
                  border: "1px solid #d1d5db",
                  fontSize: 13,
                  marginBottom: 8,
                }}
              />
              <div
                style={{
                  display: "flex",
                  justifyContent: "flex-end",
                  gap: 8,
                }}
              >
                <button
                  type="button"
                  onClick={() => {
                    setReviewForm({ rating: 5, text: "" });
                    setHoverRating(0);
                  }}
                  style={{
                    padding: "6px 10px",
                    borderRadius: 999,
                    border: "1px solid #e5e7eb",
                    background: "#fff",
                    fontSize: 13,
                    cursor: "pointer",
                  }}
                >
                  초기화
                </button>
                <button
                  type="submit"
                  disabled={reviewSubmitting}
                  style={{
                    padding: "6px 12px",
                    borderRadius: 999,
                    border: "none",
                    background: reviewSubmitting ? "#9ca3af" : "#111827",
                    color: "#fff",
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: reviewSubmitting ? "default" : "pointer",
                  }}
                >
                  {reviewSubmitting ? "등록 중..." : "리뷰 등록"}
                </button>
              </div>
            </form>

            {/* 리뷰 목록 */}
            <div
              style={{
                maxHeight: 260,
                overflowY: "auto",
                borderTop: "1px solid #e5e7eb",
                paddingTop: 8,
              }}
            >
              {reviewsLoading ? (
                <div style={{ fontSize: 13, color: "#6b7280" }}>
                  리뷰 불러오는 중...
                </div>
              ) : reviews.length === 0 ? (
                <div style={{ fontSize: 13, color: "#6b7280" }}>
                  아직 등록된 리뷰가 없어요.
                </div>
              ) : (
                reviews.map((r) => (
                  <div
                    key={r.id}
                    style={{
                      padding: "8px 0",
                      borderBottom: "1px solid #f3f4f6",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        marginBottom: 2,
                      }}
                    >
                      <div
                        style={{
                          fontSize: 12,
                          color: "#6b7280",
                        }}
                      >
                        {renderStars(r.rating)}
                        <span
                          style={{
                            marginLeft: 4,
                            fontWeight: 600,
                          }}
                        >
                          {r.rating}점
                        </span>
                      </div>
                      <div
                        style={{
                          fontSize: 11,
                          color: "#9ca3af",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {formatDateTime(r.createdAt)}
                      </div>
                    </div>
                    <div
                      style={{
                        fontSize: 13,
                        color: "#111827",
                        whiteSpace: "pre-wrap",
                      }}
                    >
                      {r.reviewText}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
