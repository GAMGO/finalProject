import React, { useEffect, useRef, useState } from "react";
import apiClient from "../api/apiClient";
import plusIcon from "../assets/plus.svg";
import plusBrown from "../assets/plus-brown.svg";
import "./KakaoMap.css";
import { favoriteApi } from "../api/apiClient";
import { useTheme } from "../theme/ThemeContext";
import { CATEGORIES } from "../constants/categories";

const APP_KEY = "bdd84bdbed2db3bc5d8b90cd6736a995";

const API_BASE = import.meta.env.VITE_BASE_URL;
const DATA_API_BASE =
  import.meta.env.VITE_BASE_URL;

const THEME_COLOR = "#78266a";

// ✅ 로그 끄기 (VITE_DEBUG=true일 때만 콘솔 찍힘)
const DEBUG = String(import.meta.env.VITE_DEBUG || "").toLowerCase() === "true";
const log = (...a) => DEBUG && console.log(...a);
const warn = (...a) => DEBUG && console.warn(...a);
const errlog = (...a) => DEBUG && console.error(...a);

// ✅ Store 객체에서 PK 꺼내는 공통 헬퍼
const getStoreIdx = (store) => {
  if (!store) return null;
  return store.idx ?? store.id ?? store.storeIdx ?? store.store_id ?? null;
};

// ✅ Store에서 lat / lng 뽑기 헬퍼
const getLatLngFromStore = (store) => {
  if (!store) return { lat: null, lng: null };
  const rawLat =
    store.latitude ??
    store.lat ??
    store.storeLatitude ??
    store.store_latitude ??
    null;
  const rawLng =
    store.longitude ??
    store.lng ??
    store.storeLongitude ??
    store.store_longitude ??
    null;

  const lat = rawLat != null ? Number(rawLat) : null;
  const lng = rawLng != null ? Number(rawLng) : null;
  return { lat, lng };
};

// ✅ 카테고리 id 가져오기
const getFoodTypeIdFromStore = (store) => {
  if (!store) return null;
  return (
    store.foodTypeId ??
    store.food_type_id ??
    store.foodTypeIdx ??
    store.food_type_idx ??
    store.categoryId ??
    store.category_id ??
    null
  );
};

// ✅ 카테고리 라벨 가져오기
const getFoodTypeLabelFromStore = (store) => {
  if (!store) return "";
  const label =
    store.foodTypeLabel ??
    store.food_type_label ??
    store.category ??
    store.categoryLabel ??
    store.category_label ??
    "";

  if (label) return label;

  const id = getFoodTypeIdFromStore(store);
  if (id == null) return "";
  return CATEGORIES.find((c) => c.id === Number(id))?.label || "";
};

// ✅ 거리 계산
const toRad = (v) => (v * Math.PI) / 180;
const distanceMeters = (lat1, lng1, lat2, lng2) => {
  const R = 6371000;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) *
      Math.cos(toRad(lat2)) *
      Math.sin(dLng / 2) *
      Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
};

// 즐겨찾기 DTO -> JS 객체
const mapFavoriteFromDto = (dto) => {
  const id = dto.id ?? dto.idx ?? dto.IDX;
  const favoriteStoreIdx =
    dto.favoriteStoreIdx ??
    dto.favorite_store_idx ??
    dto.FAVORITE_STORE_IDX ??
    null;

  return {
    id,
    favoriteStoreIdx,
    category: dto.category ?? dto.CATEGORY ?? "",
    title: dto.title ?? dto.TITLE ?? "",
    address: dto.address ?? dto.ADDRESS ?? "",
  };
};

// store -> 즐겨찾기 payload
const buildFavoritePayloadFromStore = (store) => {
  const storeIdx = getStoreIdx(store);
  const title = store.storeName || store.name || "이름 없는 노점";
  const address = store.address || store.storeAddress || "";

  const label = getFoodTypeLabelFromStore(store);
  const id = getFoodTypeIdFromStore(store);

  const category =
    label ||
    (typeof id === "number" || typeof id === "string"
      ? CATEGORIES.find((c) => c.id === Number(id))?.label
      : "기타") ||
    "기타";

  return {
    favoriteStoreIdx: storeIdx,
    category: category || "기타",
    title,
    favoriteAddress: address,
    note: "",
    rating: 0,
    imageUrl: "",
    videoUrl: "",
  };
};

export default function KakaoMap({ categoryFilterId = "" }) {
  const { theme } = useTheme();

  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const geocoderRef = useRef(null);
  const tempMarkerRef = useRef(null);

  // ✅ 전체 store 목록
  const allStoresRef = useRef([]);

  // 기본 마커
  const markersRef = useRef([]);
  // 추천 마커
  const recommendedMarkersRef = useRef([]);

  const routeLineRef = useRef(null);
  const placesRef = useRef(null);
  const myLocationMarkerRef = useRef(null);

  // 노점 등록 모달
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPos, setSelectedPos] = useState(null);
  const [form, setForm] = useState({
    categoryId: "",
    address: "",
    description: "",
  });
  const [isPickingLocation, setIsPickingLocation] = useState(false);
  const isPickingLocationRef = useRef(false);

  // 상세 + 리뷰
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [selectedStore, setSelectedStore] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [reviewStats, setReviewStats] = useState(null);
  const [reviewsLoading, setReviewsLoading] = useState(false);
  const [reviewSubmitting, setReviewSubmitting] = useState(false);
  const [reviewForm, setReviewForm] = useState({ rating: 5, text: "" });
  const [hoverRating, setHoverRating] = useState(0);

  // 리뷰 요약(AI)
  const [reviewSummary, setReviewSummary] = useState("");
  const [reviewSummaryLoading, setReviewSummaryLoading] = useState(false);
  const [reviewSummaryError, setReviewSummaryError] = useState("");

  // 즐겨찾기
  const [favorites, setFavorites] = useState([]);
  const [favoriteLoading, setFavoriteLoading] = useState(false);
  const [favoriteSaving, setFavoriteSaving] = useState(false);

  // 길찾기
  const [routeForm, setRouteForm] = useState({ from: "", to: "" });
  const [routeMode, setRouteMode] = useState("CAR");
  const [routeLoading, setRouteLoading] = useState(false);
  const [routeError, setRouteError] = useState("");

  // 내 위치
  const [myLocation, setMyLocation] = useState(null);
  const [useMyLocationAsFrom, setUseMyLocationAsFrom] = useState(false);
  const [locating, setLocating] = useState(false);

  const formatDateTime = (str) => {
    if (!str) return "";
    return str.replace("T", " ").slice(0, 16);
  };

  // 평균 별점
  const computeAvgRating = () => {
    if (
      reviewStats &&
      reviewStats.avgRating != null &&
      (reviewStats.ratingCount ?? 0) > 0
    ) {
      const n =
        typeof reviewStats.avgRating === "number"
          ? reviewStats.avgRating
          : Number(reviewStats.avgRating);
      if (!Number.isNaN(n)) return n;
    }
    if (reviews.length > 0) {
      const total = reviews.reduce((sum, r) => sum + Number(r.rating || 0), 0);
      return total / reviews.length;
    }
    return 0;
  };

  const getAvgRatingText = () => computeAvgRating().toFixed(1);

  const getRatingCount = () => {
    if (
      reviewStats &&
      typeof reviewStats.ratingCount === "number" &&
      reviewStats.ratingCount > 0
    ) {
      return reviewStats.ratingCount;
    }
    return reviews.length;
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
  // 마커 관리
  // ==========================
  const clearBaseMarkers = () => {
    markersRef.current.forEach(({ marker, infowindow }) => {
      marker.setMap(null);
      if (infowindow) infowindow.close();
    });
    markersRef.current = [];
  };

  const clearRecommendedMarkers = () => {
    recommendedMarkersRef.current.forEach(({ marker, infowindow }) => {
      marker.setMap(null);
      if (infowindow) infowindow.close();
    });
    recommendedMarkersRef.current = [];
  };

  // ==========================
  // 리뷰 + 통계
  // ==========================
  const loadReviews = async (storeIdx) => {
    if (!storeIdx) return;

    setReviewsLoading(true);
    try {
      const res = await fetch(
        `${API_BASE}/api/stores/${storeIdx}/reviews/with-stats?page=0&size=20`
      );
      const text = await res.text();

      if (!res.ok) {
        setReviews([]);
        setReviewStats(null);
        return;
      }

      const json = JSON.parse(text);
      const data = json.data ?? json;

      setReviewStats(data.stats || null);
      setReviews(Array.isArray(data.reviews) ? data.reviews : []);
    } catch (e) {
      errlog("리뷰+통계 불러오기 에러:", e);
      setReviews([]);
      setReviewStats(null);
    } finally {
      setReviewsLoading(false);
    }
  };

  // ==========================
  // 리뷰 요약(AI)
  // ==========================
  const loadReviewSummary = async (storeIdx) => {
    if (!storeIdx) return;

    setReviewSummaryLoading(true);
    setReviewSummaryError("");
    setReviewSummary("");

    try {
      const res = await fetch(`${DATA_API_BASE}/api/stores/${storeIdx}/summary`);
      const text = await res.text();

      if (!res.ok) throw new Error(`요약 요청 실패 (${res.status})`);

      const json = JSON.parse(text);
      const data = json.data ?? json;

      setReviewSummary(data.summary || "");
    } catch (e) {
      errlog("리뷰 요약 불러오기 에러:", e);
      setReviewSummaryError("리뷰 요약을 불러오지 못했어요.");
    } finally {
      setReviewSummaryLoading(false);
    }
  };

  const handleMarkerClick = (store) => {
    const storeIdx = getStoreIdx(store);
    setSelectedStore(store);
    setIsDetailOpen(true);
    setReviewForm({ rating: 5, text: "" });
    setHoverRating(0);

    loadReviews(storeIdx);
    loadReviewSummary(storeIdx);
  };

  const closeDetail = () => {
    setIsDetailOpen(false);
    setSelectedStore(null);
    setReviews([]);
    setReviewStats(null);
    setReviewForm({ rating: 5, text: "" });
    setHoverRating(0);
    setReviewSummary("");
    setReviewSummaryError("");
    setReviewSummaryLoading(false);
  };

  // ==========================
  // 가게 마커 추가
  // ==========================
  const addStoreMarker = (map, store, { recommended = false } = {}) => {
    if (!window.kakao || !map || !store) return;

    const { lat, lng } = getLatLngFromStore(store);
    if (lat == null || lng == null) return;

    const position = new window.kakao.maps.LatLng(lat, lng);

    const marker = new window.kakao.maps.Marker({
      position,
      map,
    });

    const categoryText = getFoodTypeLabelFromStore(store);
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

    if (recommended) {
      recommendedMarkersRef.current.push({ marker, infowindow });
    } else {
      markersRef.current.push({ marker, infowindow });
    }
  };

  // ✅ 필터 적용해서 base 마커 다시 그리기
  const drawBaseMarkersByFilter = (map, filterValue) => {
    if (!map) return;
    clearBaseMarkers();

    const stores = allStoresRef.current || [];
    const filterId = filterValue ? Number(filterValue) : null;

    const filtered = !filterId
      ? stores
      : stores.filter((s) => Number(getFoodTypeIdFromStore(s)) === filterId);

    filtered.forEach((s) => addStoreMarker(map, s, { recommended: false }));
  };

  const loadStoresAndDraw = async (map) => {
    try {
      const res = await fetch(`${API_BASE}/api/stores`);
      const text = await res.text();

      if (!res.ok) {
        errlog("가게 목록 불러오기 실패:", res.status);
        return;
      }

      let json;
      try {
        json = JSON.parse(text);
      } catch (e) {
        errlog("가게 목록 JSON 파싱 실패:", e);
        return;
      }

      const stores = Array.isArray(json) ? json : json.data || [];
      allStoresRef.current = stores;

      drawBaseMarkersByFilter(map, categoryFilterId);
    } catch (e) {
      errlog("가게 목록 불러오기 실패:", e);
    }
  };

  // ==========================
  // 지도 초기화
  // ==========================
  useEffect(() => {
    const scriptId = "kakao-map-sdk";

    const initMap = () => {
      if (!window.kakao || !window.kakao.maps) return;
      if (!mapRef.current) return;

      window.kakao.maps.load(async () => {
        const center = new window.kakao.maps.LatLng(37.5665, 126.978);
        const options = { center, level: 4 };

        const map = new window.kakao.maps.Map(mapRef.current, options);
        mapInstanceRef.current = map;

        geocoderRef.current = new window.kakao.maps.services.Geocoder();
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
            geocoderRef.current.coord2Address(lng, lat, (result, status) => {
              if (status === window.kakao.maps.services.Status.OK) {
                const addr =
                  result[0].road_address?.address_name ||
                  result[0].address.address_name;
                setForm((prev) => ({ ...prev, address: addr || "" }));
              }
            });
          }

          if (isPickingLocationRef.current) {
            setIsModalOpen(true);
            setIsPickingLocation(false);
            isPickingLocationRef.current = false;
          }
        });

        await loadStoresAndDraw(map);
      });
    };

    const existing = document.getElementById(scriptId);

    if (!existing) {
      const script = document.createElement("script");
      script.id = scriptId;
      script.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${APP_KEY}&autoload=false&libraries=services`;
      script.async = true;
      script.onload = () => initMap();
      script.onerror = (e) => errlog("[KAKAO] script load error", e);
      document.head.appendChild(script);
    } else {
      initMap();
    }
  }, []);

  // ✅ 사이드바 필터 변경 시: route 중 아니면 base 마커 재그리기
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    if (routeLineRef.current) return;
    drawBaseMarkersByFilter(map, categoryFilterId);
  }, [categoryFilterId]);

  // 즐겨찾기 목록 로드
  useEffect(() => {
    const loadFavorites = async () => {
      try {
        setFavoriteLoading(true);
        const list = await favoriteApi.getAll();
        const mapped = Array.isArray(list) ? list.map(mapFavoriteFromDto) : [];
        setFavorites(mapped);
      } catch (e) {
        errlog("즐겨찾기 목록 불러오기 실패", e);
      } finally {
        setFavoriteLoading(false);
      }
    };
    loadFavorites();
  }, []);

  // ==========================
  // 노점 등록 모달
  // ==========================
  const openModal = () => setIsModalOpen(true);

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

    let finalPos = selectedPos;

    if (!finalPos) {
      const addr = (form.address || "").trim();
      if (!addr) {
        alert("지도를 클릭해서 위치를 선택하거나, 주소를 입력해 주세요.");
        return;
      }

      if (!window.kakao) {
        alert("지도가 아직 준비되지 않았어요. 잠시 후 다시 시도해 주세요.");
        return;
      }

      const geocoder = geocoderRef.current;
      const places = placesRef.current;

      const searchByAddress = () =>
        new Promise((resolve, reject) => {
          if (!geocoder) return reject(new Error("지오코더가 없습니다."));
          geocoder.addressSearch(addr, (result, status) => {
            if (
              status === window.kakao.maps.services.Status.OK &&
              result &&
              result.length > 0
            ) {
              const r = result[0];
              resolve({ lat: parseFloat(r.y), lng: parseFloat(r.x) });
            } else {
              reject(new Error("주소 검색 실패"));
            }
          });
        });

      const searchByKeyword = () =>
        new Promise((resolve, reject) => {
          if (!places) return reject(new Error("장소 검색 객체가 없습니다."));
          places.keywordSearch(addr, (data, status) => {
            if (
              status === window.kakao.maps.services.Status.OK &&
              data &&
              data.length > 0
            ) {
              const d = data[0];
              resolve({ lat: parseFloat(d.y), lng: parseFloat(d.x) });
            } else {
              reject(new Error("키워드 검색 실패"));
            }
          });
        });

      try {
        try {
          finalPos = await searchByAddress();
        } catch {
          finalPos = await searchByKeyword();
        }
        setSelectedPos(finalPos);
      } catch (e2) {
        errlog("입력한 주소로 좌표 찾기 실패:", e2);
        alert(
          "입력한 주소로 위치를 찾을 수 없어요.\n지도를 클릭해서 위치를 선택해 주세요."
        );
        return;
      }
    }

    if (!finalPos) {
      alert("위치를 찾지 못했어요. 다시 시도해 주세요.");
      return;
    }

    const foodTypeId = form.categoryId ? Number(form.categoryId) : null;

    const payload = {
      storeName: form.description || "이름 없는 노점",
      foodTypeId,
      storeAddress: form.address || "",
      lat: finalPos.lat,
      lng: finalPos.lng,
    };

    try {
      const res = await fetch(`${API_BASE}/api/stores`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const text = await res.text();

      if (!res.ok) {
        alert(`가게 등록 실패 (${res.status})`);
        return;
      }

      let savedId = null;
      try {
        const json = JSON.parse(text);
        if (typeof json === "number") savedId = json;
        else if (json && typeof json === "object") {
          if (typeof json.data === "number") savedId = json.data;
          else if (typeof json.id === "number") savedId = json.id;
        }
      } catch {
        const n = Number(text);
        if (!Number.isNaN(n)) savedId = n;
      }

      const newStoreForMarker = {
        idx: savedId,
        storeName: payload.storeName,
        address: payload.storeAddress,
        latitude: payload.lat,
        longitude: payload.lng,
        foodTypeId: payload.foodTypeId,
        foodTypeLabel:
          CATEGORIES.find((c) => c.id === payload.foodTypeId)?.label || "",
      };

      allStoresRef.current = [newStoreForMarker, ...(allStoresRef.current || [])];

      if (mapInstanceRef.current) {
        if (!categoryFilterId || Number(categoryFilterId) === Number(foodTypeId)) {
          addStoreMarker(mapInstanceRef.current, newStoreForMarker, {
            recommended: false,
          });
        }
      }

      closeModal();
    } catch (e3) {
      errlog("가게 등록 실패:", e3);
      alert("가게 등록에 실패했어 ㅠㅠ");
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
      return;
    }

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
      await apiClient.post(`/api/stores/${storeIdx}/reviews`, payload);

      setReviewForm({ rating: 5, text: "" });
      setHoverRating(0);

      await loadReviews(storeIdx);
      await loadReviewSummary(storeIdx);
    } catch (e4) {
      const status = e4.response?.status;
      if (status === 401 || status === 403) {
        alert("로그인 정보가 만료되었어요. 다시 로그인해 주세요.");
      } else if (status === 400) {
        const msg =
          typeof e4.response?.data === "string"
            ? e4.response.data
            : "리뷰 내용이 정책을 위반하여 등록할 수 없습니다.";
        alert(msg);
      } else {
        alert("리뷰 등록 중 에러가 발생했어 ㅠㅠ");
      }
      errlog("리뷰 작성 에러:", status, e4.response?.data);
    } finally {
      setReviewSubmitting(false);
    }
  };

  // ==========================
  // 찜 토글
  // ==========================
  const handleToggleFavorite = async () => {
    if (!selectedStore || favoriteSaving) return;

    const storeIdx = getStoreIdx(selectedStore);
    if (!storeIdx) {
      alert("이 노점의 ID를 찾을 수 없어 찜을 저장할 수 없어요.");
      return;
    }

    const existing = favorites.find((fav) => fav.favoriteStoreIdx === storeIdx);

    try {
      setFavoriteSaving(true);

      if (!existing) {
        const payload = buildFavoritePayloadFromStore(selectedStore);
        const createdDto = await favoriteApi.create(payload);
        const created = mapFavoriteFromDto(createdDto);
        setFavorites((prev) => [...prev, created]);
      } else {
        await favoriteApi.remove(existing.id);
        setFavorites((prev) => prev.filter((f) => f.id !== existing.id));
      }
    } catch (e5) {
      errlog("찜 토글 실패", e5?.response?.status, e5?.response?.data);
      alert("찜 설정 중 오류가 발생했어요.");
    } finally {
      setFavoriteSaving(false);
    }
  };

  // ==========================
  // 내 위치
  // ==========================
  const handleUseMyLocation = () => {
    if (!navigator.geolocation) {
      setRouteError("브라우저에서 위치 정보를 지원하지 않습니다.");
      return;
    }

    setLocating(true);
    setRouteError("");

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;

        const loc = { lat, lng };
        setMyLocation(loc);
        setUseMyLocationAsFrom(true);
        setRouteForm((prev) => ({ ...prev, from: "내 위치" }));

        if (mapInstanceRef.current && window.kakao) {
          const latLng = new window.kakao.maps.LatLng(lat, lng);
          mapInstanceRef.current.setCenter(latLng);

          if (!myLocationMarkerRef.current) {
            myLocationMarkerRef.current = new window.kakao.maps.Marker({
              position: latLng,
              map: mapInstanceRef.current,
            });
          } else {
            myLocationMarkerRef.current.setPosition(latLng);
            myLocationMarkerRef.current.setMap(mapInstanceRef.current);
          }
        }

        setLocating(false);
      },
      (e) => {
        errlog("geolocation error", e);
        if (e.code === 1) {
          setRouteError("위치 권한이 거부되었습니다. 브라우저 설정을 확인해 주세요.");
        } else {
          setRouteError("내 위치를 가져오지 못했어요.");
        }
        setLocating(false);
        setUseMyLocationAsFrom(false);
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
    );
  };

  // ==========================
  // 길찾기
  // ==========================
  const handleRouteChange = (e) => {
    const { name, value } = e.target;
    if (name === "from") setUseMyLocationAsFrom(false);
    setRouteForm((prev) => ({ ...prev, [name]: value }));
  };

  const callRecommendRoute = async (startPoint, endPoint, routePoints) => {
    if (!startPoint || !endPoint) return;

    try {
      const RADIUS_M = 2000;
      const url = `${DATA_API_BASE}/recommend/route`;

      const payload = {
        start: startPoint,
        waypoints: [],
        end: endPoint,
        user_id: 10,
      };

      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const text = await res.text();

      if (!res.ok) {
        errlog("상점 추천 실패:", res.status);
        return;
      }

      let json;
      try {
        json = JSON.parse(text);
      } catch (e) {
        errlog("상점 추천 JSON 파싱 실패:", e);
        return;
      }

      const data = json.data ?? json;

      let candidates = [];
      if (Array.isArray(data.start)) candidates.push(...data.start);
      if (Array.isArray(data.end)) candidates.push(...data.end);
      if (Array.isArray(data.waypoints)) {
        data.waypoints.forEach((wp) => {
          if (Array.isArray(wp)) candidates.push(...wp);
        });
      }

      if (!mapInstanceRef.current || !window.kakao) return;

      let filtered = [];
      if (Array.isArray(routePoints) && routePoints.length) {
        filtered = candidates.filter((store) => {
          const { lat, lng } = getLatLngFromStore(store);
          if (lat == null || lng == null) return false;

          let minDist = Infinity;
          for (const p of routePoints) {
            if (p.lat == null || p.lng == null) continue;
            const d = distanceMeters(p.lat, p.lng, lat, lng);
            if (d < minDist) minDist = d;
            if (minDist <= RADIUS_M) break;
          }
          return minDist <= RADIUS_M;
        });
      } else {
        const centerLat = (startPoint.lat + endPoint.lat) / 2;
        const centerLng = (startPoint.lng + endPoint.lng) / 2;
        filtered = candidates.filter((store) => {
          const { lat, lng } = getLatLngFromStore(store);
          if (lat == null || lng == null) return false;
          const dist = distanceMeters(centerLat, centerLng, lat, lng);
          return dist <= RADIUS_M;
        });
      }

      clearRecommendedMarkers();

      if (!filtered.length) return;

      filtered.forEach((store) => {
        addStoreMarker(mapInstanceRef.current, store, { recommended: true });
      });
    } catch (e) {
      errlog("상점 추천 호출 에러:", e);
    }
  };

  const clearRoute = () => {
    setRouteForm({ from: "", to: "" });
    setRouteError("");
    setRouteLoading(false);
    setRouteMode("CAR");
    setUseMyLocationAsFrom(false);

    if (routeLineRef.current) {
      routeLineRef.current.setMap(null);
      routeLineRef.current = null;
    }

    clearRecommendedMarkers();

    if (mapInstanceRef.current) {
      drawBaseMarkersByFilter(mapInstanceRef.current, categoryFilterId);
    }
  };

  const searchLatLngByText = (raw) =>
    new Promise((resolve, reject) => {
      const keyword = (raw || "").trim();
      if (!keyword) return reject(new Error("검색어가 비어 있습니다."));

      if (!window.kakao || !window.kakao.maps || !window.kakao.maps.services) {
        return reject(new Error("카카오 지도 서비스가 준비되지 않았습니다."));
      }

      const geocoder = geocoderRef.current;
      const places = placesRef.current;
      const Status = window.kakao.maps.services.Status;

      if (geocoder) {
        geocoder.addressSearch(keyword, (result, status) => {
          if (status === Status.OK && result && result.length > 0) {
            const r = result[0];
            resolve({ lat: parseFloat(r.y), lng: parseFloat(r.x) });
          } else if (places) {
            places.keywordSearch(keyword, (data, status2) => {
              if (status2 === Status.OK && data && data.length > 0) {
                const d = data[0];
                resolve({ lat: parseFloat(d.y), lng: parseFloat(d.x) });
              } else {
                reject(new Error(`주소/장소 검색 실패: ${keyword}`));
              }
            });
          } else {
            reject(new Error("주소/장소 검색 객체가 없습니다."));
          }
        });
      } else if (places) {
        places.keywordSearch(keyword, (data, status2) => {
          if (status2 === Status.OK && data && data.length > 0) {
            const d = data[0];
            resolve({ lat: parseFloat(d.y), lng: parseFloat(d.x) });
          } else {
            reject(new Error(`주소/장소 검색 실패: ${keyword}`));
          }
        });
      } else {
        reject(new Error("주소/장소 검색 객체가 없습니다."));
      }
    });

  const handleRouteSearch = async (e) => {
    if (e) e.preventDefault();
    if (!mapInstanceRef.current || !window.kakao) return;

    const { from, to } = routeForm;

    const hasFrom =
      (from && from.trim().length > 0) || (useMyLocationAsFrom && myLocation);

    if (!hasFrom || !to) {
      setRouteError("출발지와 도착지를 모두 입력해 주세요.");
      return;
    }

    try {
      setRouteLoading(true);
      setRouteError("");

      let fromPoint;
      if (useMyLocationAsFrom && myLocation) fromPoint = myLocation;
      else fromPoint = await searchLatLngByText(from);

      const toPoint = await searchLatLngByText(to);

      clearBaseMarkers();
      clearRecommendedMarkers();

      const res = await fetch(`${API_BASE}/api/routes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          from: fromPoint,
          to: toPoint,
          mode: routeMode,
        }),
      });

      const text = await res.text();
      if (!res.ok) throw new Error(`길찾기 실패 (${res.status})`);

      const json = JSON.parse(text);
      const data = json.data ?? json;

      const points = Array.isArray(data?.path)
        ? data.path
        : Array.isArray(data?.points)
        ? data.points
        : [];

      if (!points.length) throw new Error("경로 데이터가 비어 있습니다.");

      if (routeLineRef.current) {
        routeLineRef.current.setMap(null);
        routeLineRef.current = null;
      }

      const path = points.map((p) => new window.kakao.maps.LatLng(p.lat, p.lng));

      const polyline = new window.kakao.maps.Polyline({
        path,
        strokeWeight: 5,
        strokeColor: THEME_COLOR,
        strokeOpacity: 0.9,
        strokeStyle: "solid",
      });
      polyline.setMap(mapInstanceRef.current);
      routeLineRef.current = polyline;

      const bounds = new window.kakao.maps.LatLngBounds();
      path.forEach((latlng) => bounds.extend(latlng));
      mapInstanceRef.current.setBounds(bounds);

      await callRecommendRoute(fromPoint, toPoint, points);
    } catch (e) {
      errlog("길찾기 에러:", e);
      setRouteError(e.message || "길찾기 중 에러가 발생했습니다.");

      if (mapInstanceRef.current) {
        clearRecommendedMarkers();
        drawBaseMarkersByFilter(mapInstanceRef.current, categoryFilterId);
      }
    } finally {
      setRouteLoading(false);
    }
  };

  const handleSetRouteToHere = () => {
    if (!selectedStore) return;

    const { lat, lng } = getLatLngFromStore(selectedStore);

    let placeName =
      selectedStore.address ||
      selectedStore.storeAddress ||
      selectedStore.storeName ||
      selectedStore.name ||
      "노점";

    placeName = encodeURIComponent(placeName);

    let url = "";

    if (lat != null && lng != null && !Number.isNaN(lat) && !Number.isNaN(lng)) {
      url = `https://map.kakao.com/link/to/${placeName},${lat},${lng}`;
    } else {
      const query =
        selectedStore.address ||
        selectedStore.storeAddress ||
        selectedStore.storeName ||
        "";
      if (!query) {
        alert("이 노점의 위치 정보가 없어 카카오맵을 열 수 없어요.");
        return;
      }
      url = `https://map.kakao.com/link/search/${encodeURIComponent(query)}`;
    }

    window.open(url, "_blank", "noopener,noreferrer");
  };

  // ==========================
  // 렌더
  // ==========================
  return (
    <>
      {/* 지도 */}
      <div style={{ position: "relative", width: "100vw", height: "100vh" }}>
        <div ref={mapRef} style={{ width: "100%", height: "100%" }} />
      </div>

      {/* ✅ (삭제됨) 상단 좌측 카테고리 드롭다운 UI - 사이드바로 이동 */}

      {/* 오른쪽 위 길찾기 패널 (원본 그대로) */}
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
          border: "2px solid rgba(120, 38, 106, 1)",
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: THEME_COLOR }}>
          길찾기
        </div>

        <form onSubmit={handleRouteSearch}>
          <div style={{ marginBottom: 6 }}>
            <div style={{ marginBottom: 2 }}>출발</div>
            <input
              name="from"
              value={routeForm.from}
              onChange={handleRouteChange}
              placeholder="예: 서울역 / 내 위치"
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
              placeholder="예: 시청역 / 노점 이름"
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
              justifyContent: "space-between",
              alignItems: "center",
              marginTop: 4,
              gap: 8,
            }}
          >
            <button
              type="button"
              onClick={handleUseMyLocation}
              disabled={locating}
              style={{
                borderRadius: 999,
                border: `1px solid ${THEME_COLOR}`,
                background: "#fff",
                color: THEME_COLOR,
                padding: "4px 10px",
                fontSize: 11,
                cursor: locating ? "default" : "pointer",
                whiteSpace: "nowrap",
              }}
            >
              {locating ? "위치 확인 중..." : "내 위치"}
            </button>

            <div style={{ display: "flex", justifyContent: "flex-end", gap: 6, flexShrink: 0 }}>
              <button
                type="button"
                onClick={clearRoute}
                style={{
                  borderRadius: 999,
                  border: "1px solid #e5e7eb",
                  background: "#fff",
                  padding: "4px 10px",
                  cursor: "pointer",
                  fontSize: 11,
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
                  background: routeLoading ? "#d1b5cc" : THEME_COLOR,
                  color: "#fff",
                  padding: "4px 10px",
                  fontWeight: 600,
                  cursor: routeLoading ? "default" : "pointer",
                }}
              >
                {routeLoading ? "검색 중..." : "길찾기"}
              </button>
            </div>
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
          src={theme === "dark" ? plusBrown : plusIcon}
          alt="노점 추가"
          style={{
            width: "100%",
            height: "100%",
            display: "block",
            filter: "none",
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
                <button type="button" onClick={closeModal} className="map-btn-cancel">
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
          <div className="map-modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 520 }}>
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
                {getFoodTypeLabelFromStore(selectedStore) ? (
                  <span
                    style={{
                      fontSize: 13,
                      padding: "2px 8px",
                      borderRadius: 999,
                      background: "#f3f4f6",
                      marginRight: 8,
                    }}
                  >
                    {getFoodTypeLabelFromStore(selectedStore)}
                  </span>
                ) : null}
                {selectedStore.storeName || "이름 없는 노점"}
              </h3>

              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                {(() => {
                  const storeIdx = getStoreIdx(selectedStore);
                  const isFavorited =
                    !!storeIdx && favorites.some((fav) => fav.favoriteStoreIdx === storeIdx);

                  return (
                    <button
                      type="button"
                      onClick={handleToggleFavorite}
                      disabled={favoriteSaving || favoriteLoading}
                      style={{
                        border: "none",
                        background: "transparent",
                        cursor: favoriteSaving || favoriteLoading ? "default" : "pointer",
                        fontSize: 22,
                        lineHeight: 1,
                        color: isFavorited ? THEME_COLOR : "#d1d5db",
                      }}
                      title={isFavorited ? "찜 해제" : "찜하기"}
                    >
                      {isFavorited ? "♥" : "♡"}
                    </button>
                  );
                })()}

                <button
                  type="button"
                  onClick={closeDetail}
                  style={{ border: "none", background: "transparent", fontSize: 18, cursor: "pointer" }}
                >
                  ✕
                </button>
              </div>
            </div>

            {/* 주소 */}
            {selectedStore.address || selectedStore.storeAddress ? (
              <div style={{ fontSize: 13, color: "#4b5563", marginBottom: 12 }}>
                📍 {selectedStore.address || selectedStore.storeAddress}
              </div>
            ) : null}

            {/* 평균 별점 */}
            <div
              style={{
                padding: "10px 12px",
                borderRadius: 8,
                background: "#f9fafb",
                marginBottom: 10,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <div>
                <div style={{ fontSize: 13, color: "#6b7280" }}>평균 별점</div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  {renderStars(computeAvgRating())}
                  <span style={{ fontWeight: 600, fontSize: 16 }}>{getAvgRatingText()}</span>
                  <span style={{ fontSize: 12, color: "#6b7280" }}>({getRatingCount()}개)</span>
                </div>
              </div>
            </div>

            {/* AI 리뷰 요약 */}
            <div
              style={{
                padding: "10px 12px",
                borderRadius: 8,
                background: "#fdf2ff",
                border: `1px solid ${THEME_COLOR}20`,
                marginBottom: 14,
              }}
            >
              <div style={{ fontSize: 13, color: THEME_COLOR, marginBottom: 4, fontWeight: 600 }}>
                리뷰 한 줄 요약 (AI)
              </div>
              {reviewSummaryLoading ? (
                <div style={{ fontSize: 13, color: "#6b7280" }}>요약 생성 중...</div>
              ) : reviewSummaryError ? (
                <div style={{ fontSize: 13, color: "#dc2626" }}>{reviewSummaryError}</div>
              ) : reviewSummary ? (
                <div style={{ fontSize: 13, color: "#111827", whiteSpace: "pre-wrap" }}>
                  {reviewSummary}
                </div>
              ) : (
                <div style={{ fontSize: 13, color: "#9ca3af" }}>아직 요약이 없습니다.</div>
              )}
            </div>

            {/* 리뷰 작성 */}
            <form onSubmit={handleReviewSubmit} style={{ marginBottom: 16 }}>
              <div style={{ marginBottom: 8 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <label style={{ fontSize: 13, fontWeight: 600 }}>별점</label>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    {[1, 2, 3, 4, 5].map((star) => {
                      const current = hoverRating || reviewForm.rating;
                      const filled = star <= current;
                      return (
                        <button
                          key={star}
                          type="button"
                          onClick={() => setReviewForm((prev) => ({ ...prev, rating: star }))}
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
                    <span style={{ fontSize: 13, color: "#374151", marginLeft: 4 }}>
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

              <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
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
                    background: reviewSubmitting ? "#d1b5cc" : THEME_COLOR,
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
                marginBottom: 12,
              }}
            >
              {reviewsLoading ? (
                <div style={{ fontSize: 13, color: "#6b7280" }}>리뷰 불러오는 중...</div>
              ) : reviews.length === 0 ? (
                <div style={{ fontSize: 13, color: "#6b7280" }}>아직 등록된 리뷰가 없어요.</div>
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
                      <div style={{ fontSize: 12, color: "#6b7280" }}>
                        {renderStars(r.rating)}
                        <span style={{ marginLeft: 4, fontWeight: 600 }}>{r.rating}점</span>
                      </div>
                      <div style={{ fontSize: 11, color: "#9ca3af", whiteSpace: "nowrap" }}>
                        {formatDateTime(r.createdAt)}
                      </div>
                    </div>

                    <div style={{ fontSize: 13, color: "#111827", whiteSpace: "pre-wrap" }}>
                      {r.reviewText}
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* 카카오맵 길찾기 */}
            <div style={{ display: "flex", justifyContent: "flex-start", alignItems: "center", marginTop: 4 }}>
              <button
                type="button"
                onClick={handleSetRouteToHere}
                style={{
                  padding: "6px 12px",
                  borderRadius: 999,
                  border: `1px solid ${THEME_COLOR}`,
                  background: "#fff",
                  color: THEME_COLOR,
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                카카오맵으로 길찾기
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
