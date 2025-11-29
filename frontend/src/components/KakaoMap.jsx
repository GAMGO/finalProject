// src/components/KakaoMap.jsx
import React, { useEffect, useRef, useState } from "react";
import plusIcon from "../assets/plus.svg";
import "./KakaoMap.css";

const APP_KEY = "bdd84bdbed2db3bc5d8b90cd6736a995";
const API_BASE = "http://localhost:8080"; // 뒤에 / 없음

// FOOD_INFO / FoodCategory 기준
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

export default function KakaoMap() {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const geocoderRef = useRef(null);
  const tempMarkerRef = useRef(null); // 위치 선택 중 임시 마커
  const markersRef = useRef([]); // 등록된 가게 마커들

  // ===== 노점 등록 모달 상태 =====
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPos, setSelectedPos] = useState(null); // { lat, lng }
  const [form, setForm] = useState({
    categoryId: "",
    address: "",
    description: "", // 백엔드 storeName 으로 보낼 값
  });
  const [isPickingLocation, setIsPickingLocation] = useState(false);
  const isPickingLocationRef = useRef(false);

  // ===== 가게 상세 + 리뷰 모달 상태 =====
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [selectedStore, setSelectedStore] = useState(null); // StoreResponse
  const [reviews, setReviews] = useState([]); // StoreReviewResponse[]
  const [reviewStats, setReviewStats] = useState(null); // { ratingCount, avgRating, ... }
  const [reviewsLoading, setReviewsLoading] = useState(false);
  const [reviewSubmitting, setReviewSubmitting] = useState(false);
  const [reviewForm, setReviewForm] = useState({
    rating: 5,
    text: "",
  });

  // --------------------------
  // 리뷰 불러오기
  // --------------------------
  const loadReviews = async (storeIdx) => {
    if (!storeIdx) return;
    setReviewsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/stores/${storeIdx}/reviews`);
      const text = await res.text();
      console.log("GET /api/stores/{id}/reviews:", res.status, text);

      if (!res.ok) {
        console.error("리뷰 목록 불러오기 실패:", res.status, text);
        return;
      }

      const json = JSON.parse(text);
      const data = json.data || {};
      setReviewStats(data.stats || null);
      setReviews(data.reviews || []);
    } catch (err) {
      console.error("리뷰 목록 불러오기 에러:", err);
    } finally {
      setReviewsLoading(false);
    }
  };

  // --------------------------
  // 마커 클릭 → 상세 모달 열기
  // --------------------------
  const handleMarkerClick = (store) => {
    setSelectedStore(store);
    setIsDetailOpen(true);
    setReviewForm({ rating: 5, text: "" });
    loadReviews(store.idx);
  };

  const closeDetail = () => {
    setIsDetailOpen(false);
    setSelectedStore(null);
    setReviews([]);
    setReviewStats(null);
    setReviewForm({ rating: 5, text: "" });
  };

  // --------------------------
  // 기존 가게 불러오기 + 마커
  // --------------------------
  const addStoreMarker = (map, store) => {
    if (!window.kakao || !map) return;

    const position = new window.kakao.maps.LatLng(
      store.latitude,
      store.longitude
    );

    const marker = new window.kakao.maps.Marker({
      position,
      map,
    });

    const content = `
      <div style="padding:8px 12px;font-size:12px;max-width:220px;">
        <div style="font-weight:600;margin-bottom:4px;">${store.category ?? ""}</div>
        ${
          store.storeName
            ? `<div style="margin-bottom:4px;">${store.storeName}</div>`
            : ""
        }
        ${
          store.address
            ? `<div style="font-size:11px;color:#555;">${store.address}</div>`
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

      if (!res.ok) throw new Error("load stores failed");

      const json = JSON.parse(text);
      const stores = json.data || []; // ApiResponse<List<StoreResponse>>

      stores.forEach((s) => addStoreMarker(map, s));
    } catch (err) {
      console.error("가게 목록 불러오기 실패:", err);
    }
  };

  // --------------------------
  // 지도 초기화
  // --------------------------
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

        // 주소 변환용 Geocoder
        geocoderRef.current = new window.kakao.maps.services.Geocoder();

        // 지도 클릭 시: 위치 선택 + 주소 채우기
        window.kakao.maps.event.addListener(map, "click", (mouseEvent) => {
          const latlng = mouseEvent.latLng;
          const lat = latlng.getLat();
          const lng = latlng.getLng();

          setSelectedPos({ lat, lng });

          // 임시 마커
          if (!tempMarkerRef.current) {
            tempMarkerRef.current = new window.kakao.maps.Marker({
              position: latlng,
              map,
            });
          } else {
            tempMarkerRef.current.setPosition(latlng);
          }

          // 좌표 → 주소
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

          // "지도에서 위치 선택" 모드일 때: 한 번 찍으면 모달 다시 열기
          if (isPickingLocationRef.current) {
            setIsModalOpen(true);
            setIsPickingLocation(false);
            isPickingLocationRef.current = false;
          }
        });

        console.log("[KAKAO] map created", map);

        // 기존 가게 마커들
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

  // --------------------------
  // 노점 등록 모달 open / close
  // --------------------------
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

  // 지도에서 위치 선택하기
  const handleStartPickLocation = () => {
    setIsPickingLocation(true);
    isPickingLocationRef.current = true;
    setIsModalOpen(false); // 모달 숨기고 지도 클릭 기다리기
  };

  // --------------------------
  // 입력값 변경
  // --------------------------
  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  // --------------------------
  // 노점 등록
  // --------------------------
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedPos) {
      alert("지도를 클릭해서 위치를 먼저 선택해줘요");
      return;
    }
    if (!form.categoryId) {
      alert("카테고리를 선택해줘!");
      return;
    }

    const payload = {
      storeName: form.description || "",
      foodTypeId: Number(form.categoryId),
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

      const json = JSON.parse(text);
      const saved = json.data; // ApiResponse<StoreResponse>

      if (mapInstanceRef.current) {
        addStoreMarker(mapInstanceRef.current, saved);
      }

      closeModal();
    } catch (err) {
      console.error("가게 등록 실패:", err);
      alert("가게 등록에 실패했어 ㅠㅠ 콘솔 로그 한 번 봐줘.");
    }
  };

  // --------------------------
  // 리뷰 입력 변경
  // --------------------------
  const handleReviewFormChange = (e) => {
    const { name, value } = e.target;
    setReviewForm((prev) => ({ ...prev, [name]: value }));
  };

  // --------------------------
  // 리뷰 작성
  // --------------------------
  const handleReviewSubmit = async (e) => {
    e.preventDefault();
    if (!selectedStore) return;

    const token = localStorage.getItem("jwtToken");
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
        `${API_BASE}/api/stores/${selectedStore.idx}/reviews`,
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
        alert("리뷰 등록에 실패했어 ㅠㅠ");
        return;
      }

      // 성공하면 폼 초기화 + 목록 새로 불러오기
      setReviewForm({ rating: 5, text: "" });
      await loadReviews(selectedStore.idx);
    } catch (err) {
      console.error("리뷰 작성 에러:", err);
      alert("리뷰 등록 중 에러가 발생했어 ㅠㅠ");
    } finally {
      setReviewSubmitting(false);
    }
  };

  // --------------------------
  // 별점 렌더링 유틸
  // --------------------------
  const renderStars = (value) => {
    if (!value) value = 0;
    const rounded = Math.round(value);
    return (
      <span style={{ fontSize: 18, color: "#facc15" }}>
        {[1, 2, 3, 4, 5].map((i) => (
          <span key={i}>{i <= rounded ? "★" : "☆"}</span>
        ))}
      </span>
    );
  };

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

      {/* 오른쪽 아래 + 버튼 (노점 추가) */}
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
          <div
            className="map-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="map-modal-title">노점 추가</h3>

            <form onSubmit={handleSubmit}>
              {/* 카테고리 */}
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

              {/* 주소 */}
              <label className="map-label">주소 (직접 수정 가능)</label>
              <input
                type="text"
                name="address"
                value={form.address}
                onChange={handleChange}
                placeholder="지도를 클릭하면 자동으로 채워져요"
                className="map-input"
              />

              {/* 설명 */}
              <label className="map-label">노점 설명</label>
              <textarea
                name="description"
                value={form.description}
                onChange={handleChange}
                rows={4}
                placeholder="예: 매일 저녁 7시~11시, 순살통닭/감자튀김 판매 등"
                className="map-textarea"
              />

              {/* 지도에서 위치 선택하기 버튼 */}
              <div className="map-pick-row">
                <button
                  type="button"
                  onClick={handleStartPickLocation}
                  className="map-pick-button"
                >
                  지도에서 위치 선택하기
                </button>
              </div>

              {/* 하단 버튼 */}
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

      {/* 가게 상세 + 리뷰 모달 */}
      {isDetailOpen && selectedStore && (
        <div className="map-modal-backdrop" onClick={closeDetail}>
          <div
            className="map-modal"
            onClick={(e) => e.stopPropagation()}
            style={{ maxWidth: 520 }}
          >
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
            {selectedStore.address && (
              <div
                style={{
                  fontSize: 13,
                  color: "#4b5563",
                  marginBottom: 12,
                }}
              >
                📍 {selectedStore.address}
              </div>
            )}

            {/* 평점 섹션 */}
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
                <div style={{ fontSize: 13, color: "#6b7280" }}>평균 별점</div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  {renderStars(reviewStats?.avgRating)}
                  <span style={{ fontWeight: 600, fontSize: 16 }}>
                    {reviewStats?.avgRating?.toFixed
                      ? reviewStats.avgRating.toFixed(1)
                      : reviewStats?.avgRating || "0.0"}
                  </span>
                  <span style={{ fontSize: 12, color: "#6b7280" }}>
                    ({reviewStats?.ratingCount || 0}개)
                  </span>
                </div>
              </div>
            </div>

            {/* 리뷰 작성 폼 */}
            <form onSubmit={handleReviewSubmit} style={{ marginBottom: 16 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  marginBottom: 8,
                }}
              >
                <label style={{ fontSize: 13, fontWeight: 600 }}>
                  별점
                </label>
                <select
                  name="rating"
                  value={reviewForm.rating}
                  onChange={handleReviewFormChange}
                  style={{
                    padding: "4px 8px",
                    borderRadius: 6,
                    border: "1px solid #d1d5db",
                    fontSize: 13,
                  }}
                >
                  {[5, 4, 3, 2, 1].map((v) => (
                    <option key={v} value={v}>
                      {v}점
                    </option>
                  ))}
                </select>
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
                  onClick={() => setReviewForm({ rating: 5, text: "" })}
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
                    key={r.idx}
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
                      <div style={{ fontSize: 12, color: "#6b7280" }}>
                        {renderStars(r.rating)}
                        <span style={{ marginLeft: 4, fontWeight: 600 }}>
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
                        {r.createdAt || ""}
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
