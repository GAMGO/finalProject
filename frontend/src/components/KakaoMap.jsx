// src/components/KakaoMap.jsx
import React, { useEffect, useRef, useState } from "react";
import plusIcon from "../assets/plus.svg";
import "./KakaoMap.css";

const APP_KEY = "bdd84bdbed2db3bc5d8b90cd6736a995";
const API_BASE = "http://localhost:8080/";

// FOOD_INFO.IDX 기준
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
  const tempMarkerRef = useRef(null);   // 위치 선택 중 임시 마커
  const markersRef = useRef([]);        // 등록된 가게 마커들

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPos, setSelectedPos] = useState(null); // { lat, lng }

  const [form, setForm] = useState({
    categoryId: "",
    address: "",
    description: "", // 백엔드 storeName 으로 보낼 값
  });

  // 지도에서 위치 찍는 모드 여부
  const [isPickingLocation, setIsPickingLocation] = useState(false);
  const isPickingLocationRef = useRef(false);

  // --------------------------
  // 기존 가게 불러오기
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
    });

    markersRef.current.push({ marker, infowindow });
  };

  const loadStoresAndDraw = async (map) => {
    try {
      const res = await fetch(`${API_BASE}api/stores`);
      if (!res.ok) throw new Error("load stores failed");

      const json = await res.json();
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
  // 모달 open / close
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

  // 모달 안 "지도에서 위치 선택하기" 버튼
  const handleStartPickLocation = () => {
    setIsPickingLocation(true);
    isPickingLocationRef.current = true;
    setIsModalOpen(false); // 모달 잠깐 숨기고 지도 클릭 기다리기
  };

  // --------------------------
  // 입력값 변경
  // --------------------------
  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  // --------------------------
  // 가게 등록
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
      const res = await fetch(`${API_BASE}/stores`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("create store failed");

      const json = await res.json();
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

  return (
    <>
      {/* 지도 박스 */}
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

      {/* 오른쪽 아래 + 버튼 */}
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
        <div
          className="map-modal-backdrop"
          onClick={closeModal}
        >
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
                <button
                  type="submit"
                  className="map-btn-submit"
                >
                  등록하기
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
