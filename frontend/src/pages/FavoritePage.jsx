// src/pages/FavoritePage.jsx
import React, { useMemo, useState, useEffect } from "react";
import "./FavoritePage.css";
import { favoriteApi } from "../api/apiClient";
import MediaEmbed from "../components/MediaEmbed";

// ===== 아이콘 =====
import allIcon from "../assets/favIcons/All.png";
import chickenIcon from "../assets/favIcons/chicken.png";
import ddeokbokkiIcon from "../assets/favIcons/ddeokbokki.png";
import bungebbangIcon from "../assets/favIcons/bungebbang.png";
import seafoodIcon from "../assets/favIcons/seafood.png";
import takoyakiIcon from "../assets/favIcons/takoyaki.png";

import bbeongtIcon from "../assets/favIcons/bbeongttuigi.png";
import chestnutSweetpotatoIcon from "../assets/favIcons/chestnut_sweatpotato.png";
import cornIcon from "../assets/favIcons/corn.png";
import eggbreadIcon from "../assets/favIcons/eggbread.png";
import skewersIcon from "../assets/favIcons/skewers.png";
import sundaeGopchangIcon from "../assets/favIcons/sundae_gopchang.png";

import etcIcon from "../assets/favIcons/Etc.png";

// ===== 샘플 사진 (fallback 용) =====
import FAV_CHICKEN from "../assets/images/favorites/favChicken.jpg";
import FAV_BUNGEOPPANG from "../assets/images/favorites/favBungeoppang.jpg";
import FAV_BUSNIK from "../assets/images/favorites/favBunsik.jpg";
import FAV_TAKOYAKI from "../assets/images/favorites/favTakoyaki.png";
import FAV_SUNDAE_GOPCHANG from "../assets/images/favorites/favSundaeGopchang.jpg";
import FAV_SEAFOOD from "../assets/images/favorites/favSeafood.jpg";
import FAV_BBEONGTTEUGI from "../assets/images/favorites/favBbeongtteugi.jpg";
import FAV_EGG_BREAD from "../assets/images/favorites/favEggBread.jpg";
import FAV_CORN from "../assets/images/favorites/favCorn.jpg";
import FAV_GUNBAM_GOGUMA from "../assets/images/favorites/favGunbamGoguma.png";
import FAV_SKEWERS from "../assets/images/favorites/favSkewers.png";

/** 폼에서 카테고리 선택용(기존 그대로 유지) */
const FILTERS = [
  { key: "전체", label: "전체", icon: allIcon },
  { key: "통닭", label: "통닭", icon: chickenIcon },
  { key: "타코야끼", label: "타코야끼", icon: takoyakiIcon },
  { key: "순대곱창", label: "순대·곱창", icon: sundaeGopchangIcon },
  { key: "붕어빵", label: "붕어빵", icon: bungebbangIcon },
  { key: "군밤/고구마", label: "군밤/고구마", icon: chestnutSweetpotatoIcon },
  { key: "닭꼬치", label: "닭꼬치", icon: skewersIcon },
  { key: "분식", label: "분식", icon: ddeokbokkiIcon },
  { key: "해산물", label: "해산물", icon: seafoodIcon },
  { key: "뻥튀기", label: "뻥튀기", icon: bbeongtIcon },
  { key: "계란빵", label: "계란빵", icon: eggbreadIcon },
  { key: "옥수수", label: "옥수수", icon: cornIcon },
  { key: "기타", label: "기타", icon: etcIcon },
];

const CATEGORY_ALIAS = {
  분식: ["분식", "떡볶이"],
};

const DEFAULT_CROP = {
  offsetX: 50,
  offsetY: 50,
  zoom: 1,
};

const DEFAULT_CARD_IMAGE = {
  통닭: FAV_CHICKEN,
  타코야끼: FAV_TAKOYAKI,
  순대곱창: FAV_SUNDAE_GOPCHANG,
  붕어빵: FAV_BUNGEOPPANG,
  "군밤/고구마": FAV_GUNBAM_GOGUMA,
  닭꼬치: FAV_SKEWERS,
  분식: FAV_BUSNIK,
  떡볶이: FAV_BUSNIK,
  해산물: FAV_SEAFOOD,
  뻥튀기: FAV_BBEONGTTEUGI,
  계란빵: FAV_EGG_BREAD,
  옥수수: FAV_CORN,
};

const getFallbackImage = (category) => {
  if (DEFAULT_CARD_IMAGE[category]) return DEFAULT_CARD_IMAGE[category];

  for (const [base, aliases] of Object.entries(CATEGORY_ALIAS)) {
    if (aliases.includes(category)) {
      return DEFAULT_CARD_IMAGE[base] || "";
    }
  }
  return "";
};

const mapFromDto = (dto) => {
  const id = dto.id ?? dto.idx ?? dto.IDX;

  const category = dto.category ?? dto.CATEGORY ?? "기타";
  const title = dto.title ?? dto.TITLE ?? "";
  const favoriteAddress =
    dto.favoriteAddress ?? dto.FAVORITE_ADDRESS ?? dto.address ?? "";
  const note = dto.note ?? dto.NOTE ?? "";
  const ratingRaw = dto.rating ?? dto.RATING ?? 0;
  const rating = typeof ratingRaw === "number" ? ratingRaw : Number(ratingRaw) || 0;

  const imageUrl = dto.imageUrl ?? dto.IMAGE_URL ?? "";
  const videoUrl = dto.videoUrl ?? dto.VIDEO_URL ?? "";

  const fallbackImage = getFallbackImage(category);

  return {
    id,
    category,
    title,
    address: favoriteAddress,
    note,
    rating,
    image: imageUrl || fallbackImage,
    videoUrl,
    createdAt: dto.createdAt ?? dto.CREATED_AT ?? null,
    expiredAt: dto.expiredAt ?? dto.EXPIRED_AT ?? null,
  };
};

export default function FavoritePage({ categoryFilter = "전체" }) {
  // ✅ 사이드바에서 내려온 값만 사용
  const filter = categoryFilter || "전체";

  const [favorites, setFavorites] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const [cropState, setCropState] = useState({});
  const [editingCropId, setEditingCropId] = useState(null);
  const [draftCrop, setDraftCrop] = useState({ ...DEFAULT_CROP });

  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({
    x: 0,
    y: 0,
    offsetX: 50,
    offsetY: 50,
  });

  const [isFormOpen, setIsFormOpen] = useState(false);
  const [formState, setFormState] = useState({
    id: null,
    category: "통닭",
    title: "",
    address: "",
    note: "",
    rating: 4.5,
    imageUrl: "",
    videoUrl: "",
    customCategoryRequest: "",
  });

  const [isSaving, setIsSaving] = useState(false);
  const [hoverRating, setHoverRating] = useState(null);

  const fetchFavorites = async () => {
    try {
      setIsLoading(true);
      const list = await favoriteApi.getAll();
      const mapped = Array.isArray(list) ? list.map(mapFromDto) : [];
      setFavorites(mapped);
    } catch (error) {
      console.error("즐겨찾기 목록 불러오기 실패", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchFavorites();
  }, []);

  useEffect(() => {
    setCropState((prev) => {
      const next = { ...prev };
      favorites.forEach((fav) => {
        if (!next[fav.id]) next[fav.id] = { ...DEFAULT_CROP };
      });
      return next;
    });
  }, [favorites]);

  const filteredFavorites = useMemo(() => {
    if (filter === "전체") return favorites;

    if (CATEGORY_ALIAS[filter]) {
      const aliases = CATEGORY_ALIAS[filter];
      return favorites.filter((f) => aliases.includes(f.category));
    }

    return favorites.filter((f) => f.category === filter);
  }, [favorites, filter]);

  const openCropFor = (favId) => {
    const base = cropState[favId] || { ...DEFAULT_CROP };
    setEditingCropId(favId);
    setDraftCrop({ ...base });
  };

  const handleCropCancel = () => {
    setEditingCropId(null);
    setDraftCrop({ ...DEFAULT_CROP });
    setIsDragging(false);
  };

  const handleCropSave = () => {
    if (!editingCropId) return;
    setCropState((prev) => ({
      ...prev,
      [editingCropId]: { ...draftCrop },
    }));
    setEditingCropId(null);
    setIsDragging(false);
  };

  const handleCropReset = () => setDraftCrop({ ...DEFAULT_CROP });

  const handleZoomChange = (value) => {
    const v = Math.max(1, Math.min(2, value));
    setDraftCrop((prev) => ({ ...prev, zoom: v }));
  };

  const handleZoomStep = (delta) => {
    setDraftCrop((prev) => {
      const next = Math.max(1, Math.min(2, prev.zoom + delta));
      return { ...prev, zoom: next };
    });
  };

  const handleCropMouseDown = (event) => {
    event.preventDefault();
    const container = event.currentTarget.getBoundingClientRect();
    setIsDragging(true);
    setDragStart({
      x: event.clientX,
      y: event.clientY,
      offsetX: draftCrop.offsetX,
      offsetY: draftCrop.offsetY,
      width: container.width,
      height: container.height,
    });
  };

  useEffect(() => {
    if (!isDragging) return;

    const handleMove = (event) => {
      setDraftCrop((prev) => {
        const dx = event.clientX - dragStart.x;
        const dy = event.clientY - dragStart.y;

        const moveX =
          dragStart.width && dragStart.width > 0 ? (dx / dragStart.width) * 100 : 0;
        const moveY =
          dragStart.height && dragStart.height > 0 ? (dy / dragStart.height) * 100 : 0;

        let nextX = dragStart.offsetX + moveX;
        let nextY = dragStart.offsetY + moveY;

        nextX = Math.max(0, Math.min(100, nextX));
        nextY = Math.max(0, Math.min(100, nextY));

        return { ...prev, offsetX: nextX, offsetY: nextY };
      });
    };

    const handleUp = () => setIsDragging(false);

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);

    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [isDragging, dragStart]);

  const handleUnfavorite = async (id) => {
    if (!window.confirm("이 즐겨찾기를 해제할까요?")) return;

    try {
      await favoriteApi.remove(id);
      setFavorites((prev) => prev.filter((f) => f.id !== id));
      setCropState((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    } catch (error) {
      console.error("즐겨찾기 해제 실패", error);
      alert("즐겨찾기 해제에 실패했습니다.");
    }
  };

  const openEditForm = (fav) => {
    setFormState({
      id: fav.id,
      category: fav.category,
      title: fav.title,
      address: fav.address,
      note: fav.note || "",
      rating: fav.rating ?? 4.5,
      imageUrl: fav.image || "",
      videoUrl: fav.videoUrl || "",
      customCategoryRequest: "",
    });
    setHoverRating(null);
    setIsFormOpen(true);
  };

  const handleFormChange = (field, value) => {
    setFormState((prev) => ({ ...prev, [field]: value }));
  };

  const handleFormImageChange = (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);

    if (file.type.startsWith("video/")) {
      setFormState((prev) => ({ ...prev, imageUrl: "", videoUrl: url }));
    } else {
      setFormState((prev) => ({ ...prev, imageUrl: url }));
    }
  };

  const getHalfOrFullValue = (starIndex, event) => {
    const native = event.nativeEvent;
    const target = native.target;
    const width = target.clientWidth || 1;
    const offsetX = native.offsetX;
    const isHalf = offsetX < width / 2;
    return isHalf ? starIndex - 0.5 : starIndex;
  };

  const handleStarClick = (starIndex, event) => {
    const value = getHalfOrFullValue(starIndex, event);
    handleFormChange("rating", value);
  };

  const handleStarHover = (starIndex, event) => {
    const value = getHalfOrFullValue(starIndex, event);
    setHoverRating(value);
  };

  const renderFormStars = () => {
    const activeRating = hoverRating != null ? hoverRating : formState.rating || 0;

    return (
      <div className="fav-form-rating">
        {[1, 2, 3, 4, 5].map((star) => {
          let cls = "empty";
          if (activeRating >= star) cls = "full";
          else if (activeRating >= star - 0.5) cls = "half";

          return (
            <button
              key={star}
              type="button"
              className={`fav-star ${cls}`}
              onClick={(event) => handleStarClick(star, event)}
              onMouseMove={(event) => handleStarHover(star, event)}
              onMouseLeave={() => setHoverRating(null)}
            >
              ★
            </button>
          );
        })}
        <span className="fav-form-rating-score">
          {formState.rating ? formState.rating.toFixed(1) : "0.0"}
        </span>
      </div>
    );
  };

  const renderStaticStars = (rating) => {
    const value = rating || 0;
    return (
      <span className="fav-static-stars">
        {[1, 2, 3, 4, 5].map((star) => {
          let cls = "empty";
          if (value >= star) cls = "full";
          else if (value >= star - 0.5) cls = "half";
          return (
            <span key={star} className={`fav-star-static ${cls}`}>
              ★
            </span>
          );
        })}
      </span>
    );
  };

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    if (isSaving) return;

    const trimmedTitle = formState.title.trim();
    if (!trimmedTitle) return;

    const baseImage =
      formState.imageUrl ||
      (favorites.find((f) => f.id === formState.id)?.image ||
        getFallbackImage(formState.category));

    const trimmedVideoUrl =
      typeof formState.videoUrl === "string"
        ? formState.videoUrl.trim()
        : formState.videoUrl || "";

    const payload = {
      idx: formState.id ?? null,
      id: formState.id ?? null,
      customer_idx: 1,
      favoriteStoreIdx: null,

      category: formState.category,
      title: trimmedTitle,
      favoriteAddress: formState.address.trim(),
      note: formState.note.trim(),
      rating: Number(formState.rating) || 0,
      imageUrl: baseImage,
      videoUrl: trimmedVideoUrl || "",
    };

    try {
      setIsSaving(true);
      const updatedDto = await favoriteApi.update(formState.id, payload);
      const updated = mapFromDto(updatedDto);
      setFavorites((prev) => prev.map((fav) => (fav.id === updated.id ? updated : fav)));

      if (formState.customCategoryRequest.trim()) {
        console.log("새 카테고리 요청:", formState.customCategoryRequest.trim());
      }

      setIsFormOpen(false);
    } catch (error) {
      console.error("즐겨찾기 저장 실패", error);
      alert("즐겨찾기 저장에 실패했습니다.");
    } finally {
      setIsSaving(false);
    }
  };

  const handleFormCancel = () => {
    if (isSaving) return;
    setIsFormOpen(false);
  };

  return (
    <div className="favorite-root">
      {/* ✅ 상단 헤더(카테고리 줄) 완전 제거 */}

      <div className="favorite-inner">
        {isLoading ? (
          <div className="fav-empty">즐겨찾기를 불러오는 중입니다...</div>
        ) : filteredFavorites.length === 0 ? (
          <div className="fav-empty">즐겨찾기한 노점이 없어요.</div>
        ) : (
          <div className="fav-list">
            {filteredFavorites.map((item) => {
              const crop = cropState[item.id] || DEFAULT_CROP;

              const imgStyle = {
                "--crop-x": `${crop.offsetX}%`,
                "--crop-y": `${crop.offsetY}%`,
                "--crop-zoom": crop.zoom,
              };

              const hasVideo = !!item.videoUrl;

              return (
                <article key={item.id} className="fav-card">
                  <div className="fav-card-header">
                    <div className="fav-card-badge">{item.category}</div>
                    <div className="fav-card-header-right">
                      <button
                        type="button"
                        className="fav-edit"
                        onClick={() => openEditForm(item)}
                      >
                        수정
                      </button>
                      <button
                        type="button"
                        className="fav-unlike"
                        onClick={() => handleUnfavorite(item.id)}
                      >
                        해제
                      </button>
                    </div>
                  </div>

                  <div className="fav-card-image-wrap">
                    {hasVideo ? (
                      <MediaEmbed
                        url={item.videoUrl}
                        poster={item.image}
                        className="fav-card-media"
                      />
                    ) : item.image ? (
                      <>
                        <img
                          src={item.image}
                          alt={item.title}
                          className="fav-card-image"
                          style={imgStyle}
                        />
                        <button
                          type="button"
                          className="fav-image-edit-btn"
                          onClick={() => openCropFor(item.id)}
                        >
                          초점 조절
                        </button>
                      </>
                    ) : (
                      <div className="fav-card-noimage">사진 없음</div>
                    )}
                  </div>

                  <div className="fav-card-body">
                    <div className="fav-card-body-main">
                      <div className="fav-card-text">
                        <h3 className="fav-card-title" onClick={() => openEditForm(item)}>
                          {item.title}
                        </h3>
                        <div className="fav-card-addr">📍 {item.address}</div>
                      </div>

                      {item.note && <p className="fav-card-note">{item.note}</p>}

                      {typeof item.rating === "number" && (
                        <div className="fav-card-rating fav-card-rating-right">
                          {renderStaticStars(item.rating)}
                          <span className="fav-card-rating-score">
                            {item.rating.toFixed(1)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </div>

      {editingCropId && (
        <div className="fav-crop-modal-backdrop">
          <div className="fav-crop-modal">
            <div className="fav-crop-modal-header">
              <span>사진 위치 조정</span>
              <small>아래 화면이 실제 카드에 적용되는 모습과 100% 동일합니다.</small>
            </div>

            <div className="fav-crop-frame" onMouseDown={handleCropMouseDown}>
              <img
                src={favorites.find((f) => f.id === editingCropId)?.image || ""}
                alt="crop"
                className="fav-card-image"
                style={{
                  objectPosition: `${draftCrop.offsetX}% ${draftCrop.offsetY}%`,
                  transform: `scale(${draftCrop.zoom})`,
                }}
              />
              <div className="fav-crop-grid" />
            </div>

            <div className="fav-crop-controls">
              <div className="fav-crop-zoom-row">
                <button
                  type="button"
                  className="fav-btn ghost small"
                  onClick={() => handleZoomStep(-0.1)}
                >
                  -
                </button>
                <input
                  type="range"
                  min="100"
                  max="200"
                  value={Math.round(draftCrop.zoom * 100)}
                  onChange={(event) => handleZoomChange(Number(event.target.value) / 100)}
                />
                <button
                  type="button"
                  className="fav-btn ghost small"
                  onClick={() => handleZoomStep(0.1)}
                >
                  +
                </button>
              </div>
              <div className="fav-crop-tip">
                마우스로 사진을 드래그해서 위치를 옮길 수 있어요.
              </div>
            </div>

            <div className="fav-crop-modal-actions">
              <button type="button" className="fav-btn ghost" onClick={handleCropCancel}>
                취소
              </button>
              <button type="button" className="fav-btn ghost" onClick={handleCropReset}>
                원본으로
              </button>
              <button type="button" className="fav-btn primary" onClick={handleCropSave}>
                완료
              </button>
            </div>
          </div>
        </div>
      )}

      {isFormOpen && (
        <div className="fav-form-backdrop">
          <form className="fav-form" onSubmit={handleFormSubmit}>
            <div className="fav-form-header">
              <h3 className="fav-form-title">즐겨찾기 수정</h3>
              <p className="fav-form-subtitle">지나가다 본 노점, 기억날 때 후딱 수정해두자.</p>
            </div>

            <div className="fav-form-section">
              <div className="fav-form-field">
                <label>사진 / 영상</label>
                <div className="fav-form-image-input">
                  {formState.imageUrl && (
                    <img
                      src={formState.imageUrl}
                      alt="미리보기"
                      className="fav-form-image-preview"
                    />
                  )}
                  {!formState.imageUrl &&
                    formState.videoUrl &&
                    formState.videoUrl.startsWith("blob:") && (
                      <video
                        src={formState.videoUrl}
                        className="fav-form-video-preview"
                        controls
                      />
                    )}
                  <label className="fav-file-label">
                    파일 선택
                    <input
                      type="file"
                      accept="image/*,video/*"
                      onChange={handleFormImageChange}
                      hidden
                    />
                  </label>
                </div>
              </div>

              <div className="fav-form-field">
                <label>온라인 영상 링크 (선택)</label>
                <input
                  type="text"
                  placeholder="예: 유튜브 / 네이버 / 카카오 등 영상 주소"
                  value={formState.videoUrl.startsWith("blob:") ? "" : formState.videoUrl}
                  onChange={(event) => handleFormChange("videoUrl", event.target.value)}
                />
                <small>
                  온라인 영상 주소를 붙여넣으면 카드에서 바로 재생을 시도해요.
                </small>
              </div>

              <div className="fav-form-field">
                <label>카테고리</label>
                <select
                  value={formState.category}
                  onChange={(event) => handleFormChange("category", event.target.value)}
                >
                  {FILTERS.filter((f) => f.key !== "전체").map((f) => (
                    <option key={f.key} value={f.key}>
                      {f.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="fav-form-field">
                <label>새 카테고리 요청 (선택)</label>
                <input
                  type="text"
                  placeholder="예: 어묵, 붕어빵+아이스크림 등"
                  value={formState.customCategoryRequest}
                  onChange={(event) =>
                    handleFormChange("customCategoryRequest", event.target.value)
                  }
                />
                <small>기존 분류에 없으면 여기 적어서 관리자에게 요청.</small>
              </div>
            </div>

            <div className="fav-form-section">
              <div className="fav-form-field">
                <label>상호 / 이름</label>
                <input
                  type="text"
                  placeholder="예: 시청 앞 통닭 트럭"
                  value={formState.title}
                  onChange={(event) => handleFormChange("title", event.target.value)}
                  required
                />
              </div>

              <div className="fav-form-field">
                <label>위치</label>
                <input
                  type="text"
                  placeholder="예: ○○역 3번 출구 앞"
                  value={formState.address}
                  onChange={(event) => handleFormChange("address", event.target.value)}
                />
              </div>

              <div className="fav-form-field">
                <label>한줄 설명</label>
                <textarea
                  rows={3}
                  placeholder="예: 줄 길지만 진짜 맛있음"
                  value={formState.note}
                  onChange={(event) => handleFormChange("note", event.target.value)}
                />
              </div>
            </div>

            <div className="fav-form-section">
              <div className="fav-form-field">
                <label>평점</label>
                {renderFormStars()}
                <small>별 끝을 클릭하면 0.5단위로 조절할 수 있어요.</small>
              </div>
            </div>

            <div className="fav-form-actions">
              <button
                type="button"
                className="fav-btn ghost"
                onClick={handleFormCancel}
                disabled={isSaving}
              >
                취소
              </button>
              <button type="submit" className="fav-btn primary" disabled={isSaving}>
                {isSaving ? "저장 중..." : "저장"}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
