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

// 새로 만든 아이콘들
import bbeongtIcon from "../assets/favIcons/bbeongttuigi.png"; // 뻥튀기
import chestnutSweetpotatoIcon from "../assets/favIcons/chestnut_sweatpotato.png"; // 군밤/고구마
import cornIcon from "../assets/favIcons/corn.png"; // 옥수수
import eggbreadIcon from "../assets/favIcons/eggbread.png"; // 계란빵
import skewersIcon from "../assets/favIcons/skewers.png"; // 닭꼬치
import sundaeGopchangIcon from "../assets/favIcons/sundae_gopchang.png"; // 순대곱창

// 기타 공용
import etcIcon from "../assets/favIcons/Etc.png";

// ===== 샘플 사진 (fallback 용) =====
import favChicken from "../assets/images/favorites/chicken.jpg";
import favBungebbang from "../assets/images/favorites/bungebbang.jpg";
import favPojangmacha from "../assets/images/favorites/pojangmacha.jpg";

/** KakaoMap 쪽 드롭다운이랑 맞춰 놓은 카테고리들 */
const FILTERS = [
  { key: "전체", label: "전체", icon: allIcon, iconScale: 1.7 },

  { key: "통닭", label: "통닭", icon: chickenIcon, iconScale: 1.7 },
  { key: "타코야끼", label: "타코야끼", icon: takoyakiIcon, iconScale: 1.7 },
  { key: "순대곱창", label: "순대·곱창", icon: sundaeGopchangIcon, iconScale: 1.7 },
  { key: "붕어빵", label: "붕어빵", icon: bungebbangIcon, iconScale: 1.7 },
  { key: "분식", label: "분식", icon: ddeokbokkiIcon, iconScale: 1.7 },

  { key: "해산물", label: "해산물", icon: seafoodIcon, iconScale: 1.7 },
  { key: "뻥튀기", label: "뻥튀기", icon: bbeongtIcon, iconScale: 1.7 },
  { key: "계란빵", label: "계란빵", icon: eggbreadIcon, iconScale: 1.7 },
  { key: "옥수수", label: "옥수수", icon: cornIcon, iconScale: 1.7 },
  {
    key: "군밤/고구마",
    label: "군밤/고구마",
    icon: chestnutSweetpotatoIcon,
    iconScale: 1.7,
  },
  { key: "닭꼬치", label: "닭꼬치", icon: skewersIcon, iconScale: 1.7 },

  { key: "기타", label: "기타", icon: etcIcon, iconScale: 1.7 },
];

const CATEGORY_ALIAS = {
  분식: ["분식", "떡볶이"],
};

const DEFAULT_CROP = {
  offsetX: 50,
  offsetY: 50,
  zoom: 1,
};

const getFallbackImage = (category) => {
  switch (category) {
    case "붕어빵":
      return favBungebbang;
    case "통닭":
      return favChicken;
    default:
      return favPojangmacha;
  }
};

// 숫자 id 변환 유틸
const toNumericId = (raw) => {
  const n =
    typeof raw === "number"
      ? raw
      : raw != null
      ? Number(raw)
      : NaN;
  return Number.isFinite(n) && n > 0 ? n : null;
};

// 백엔드 DTO -> 프론트에서 쓰는 형태로 매핑
const mapFromDto = (dto) => {
  const id = toNumericId(dto.id ?? dto.idx ?? dto.IDX);

  const category = dto.category ?? dto.CATEGORY ?? "기타";
  const title = dto.title ?? dto.TITLE ?? "";
  const favoriteAddress =
    dto.favoriteAddress ?? dto.FAVORITE_ADDRESS ?? dto.address ?? "";
  const note = dto.note ?? dto.NOTE ?? "";
  const ratingRaw = dto.rating ?? dto.RATING ?? 0;
  const rating =
    typeof ratingRaw === "number" ? ratingRaw : Number(ratingRaw) || 0;

  const imageUrl = dto.imageUrl ?? dto.IMAGE_URL ?? "";
  const videoUrl = dto.videoUrl ?? dto.VIDEO_URL ?? "";

  return {
    id,
    category,
    title,
    address: favoriteAddress,
    note,
    rating,
    image: imageUrl || getFallbackImage(category),
    videoUrl,
    createdAt: dto.createdAt ?? dto.CREATED_AT ?? null,
    expiredAt: dto.expiredAt ?? dto.EXPIRED_AT ?? null,
  };
};

export default function FavoritePage() {
  // 카테고리 필터
  const [filter, setFilter] = useState("전체");

  // 즐겨찾기 목록 (실제 화면용 오브젝트)
  const [favorites, setFavorites] = useState([]);

  // DB 로딩 상태
  const [isLoading, setIsLoading] = useState(false);

  // 이미지 초점 조절 상태: { [id]: { offsetX, offsetY, zoom } }
  const [cropState, setCropState] = useState({});

  // 크롭 모달 상태
  const [editingCropId, setEditingCropId] = useState(null);
  const [draftCrop, setDraftCrop] = useState({ ...DEFAULT_CROP });

  // 드래그 상태
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({
    x: 0,
    y: 0,
    offsetX: 50,
    offsetY: 50,
  });

  // 등록/수정 폼
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [formState, setFormState] = useState({
    id: null,
    category: "통닭",
    title: "",
    address: "",
    note: "",
    rating: 4.5,
    imageUrl: "",
    videoUrl: "", // 파일/온라인 공용
    customCategoryRequest: "",
  });

  // 저장 중 상태 (중복 저장 방지)
  const [isSaving, setIsSaving] = useState(false);

  // 별점 호버 상태 (.5 단위)
  const [hoverRating, setHoverRating] = useState(null);

  // ====== DB에서 즐겨찾기 목록 불러오기 ======
  const fetchFavorites = async () => {
    try {
      setIsLoading(true);
      const list = await favoriteApi.getAll(); // FavoriteResponse[] or ApiResponse.data
      const mapped = Array.isArray(list) ? list.map(mapFromDto) : [];
      setFavorites(mapped);
    } catch (error) {
      console.error("즐겨찾기 목록 불러오기 실패", error);
    } finally {
      setIsLoading(false);
    }
  };

  // 마운트 시 1회 로딩
  useEffect(() => {
    fetchFavorites();
  }, []);

  // favorites가 바뀔 때마다 새로 추가된 것들에 기본 크롭값 세팅
  useEffect(() => {
    setCropState((prev) => {
      const next = { ...prev };
      favorites.forEach((fav) => {
        if (!next[fav.id]) {
          next[fav.id] = { ...DEFAULT_CROP };
        }
      });
      return next;
    });
  }, [favorites]);

  // 필터링 된 목록
  const filteredFavorites = useMemo(() => {
    if (filter === "전체") return favorites;

    if (CATEGORY_ALIAS[filter]) {
      const aliases = CATEGORY_ALIAS[filter];
      return favorites.filter((f) => aliases.includes(f.category));
    }

    return favorites.filter((f) => f.category === filter);
  }, [favorites, filter]);

  // ===== 이미지 크롭 모달 열기 =====
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

  const handleCropReset = () => {
    setDraftCrop({ ...DEFAULT_CROP });
  };

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

  // 드래그 시작
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

  // 전역 드래그 처리
  useEffect(() => {
    if (!isDragging) return;

    const handleMove = (event) => {
      setDraftCrop((prev) => {
        const dx = event.clientX - dragStart.x;
        const dy = event.clientY - dragStart.y;

        const moveX =
          dragStart.width && dragStart.width > 0
            ? (dx / dragStart.width) * 100
            : 0;
        const moveY =
          dragStart.height && dragStart.height > 0
            ? (dy / dragStart.height) * 100
            : 0;

        let nextX = dragStart.offsetX + moveX;
        let nextY = dragStart.offsetY + moveY;

        nextX = Math.max(0, Math.min(100, nextX));
        nextY = Math.max(0, Math.min(100, nextY));

        return { ...prev, offsetX: nextX, offsetY: nextY };
      });
    };

    const handleUp = () => {
      setIsDragging(false);
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);

    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [isDragging, dragStart]);

  // ===== 즐겨찾기 삭제 (DB + 화면) =====
  const handleUnfavorite = async (id) => {
    if (!window.confirm("이 즐겨찾기를 해제할까요?")) return;

    const numericId = toNumericId(id);
    if (numericId == null) {
      console.error("잘못된 즐겨찾기 id:", id);
      return;
    }

    try {
      await favoriteApi.remove(numericId);
      setFavorites((prev) => prev.filter((f) => f.id !== numericId));
      setCropState((prev) => {
        const next = { ...prev };
        delete next[numericId];
        return next;
      });
    } catch (error) {
      console.error("즐겨찾기 해제 실패", error);
      alert("즐겨찾기 해제에 실패했습니다.");
    }
  };

  // ===== 등록/수정 폼 =====
  const openNewForm = () => {
    setFormState({
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
    setHoverRating(null);
    setIsFormOpen(true);
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
      setFormState((prev) => ({
        ...prev,
        imageUrl: "",
        videoUrl: url,
      }));
    } else {
      setFormState((prev) => ({
        ...prev,
        imageUrl: url,
      }));
    }
  };

  // 별 클릭 시 .5 / 1.0 결정
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
    const activeRating =
      hoverRating != null ? hoverRating : formState.rating || 0;

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

  // ===== 폼 제출 → DB 저장(Create/Update) =====
  const handleFormSubmit = async (event) => {
    event.preventDefault();
    if (isSaving) return;

    const trimmedTitle = formState.title.trim();
    if (!trimmedTitle) return;

    const numericId = toNumericId(formState.id);

    const baseImage =
      formState.imageUrl ||
      (numericId == null
        ? getFallbackImage(formState.category)
        : favorites.find((f) => f.id === numericId)?.image ||
          getFallbackImage(formState.category));

    const trimmedVideoUrl =
      typeof formState.videoUrl === "string"
        ? formState.videoUrl.trim()
        : formState.videoUrl || "";

    const payload = {
      idx: numericId,
      id: numericId,
      customerIdx: 1, // 지금은 서비스에서 1L 쓰고 있다 했으니 그대로 둠
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

      if (numericId == null) {
        // 새 등록
        const createdDto = await favoriteApi.create(payload);
        const created = mapFromDto(createdDto);
        setFavorites((prev) => [...prev, created]);
      } else {
        // 수정
        const updatedDto = await favoriteApi.update(numericId, payload);
        const updated = mapFromDto(updatedDto);
        setFavorites((prev) =>
          prev.map((fav) => (fav.id === updated.id ? updated : fav))
        );
      }

      if (formState.customCategoryRequest.trim()) {
        console.log(
          "새 카테고리 요청:",
          formState.customCategoryRequest.trim()
        );
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

  // 필터 칩 2행으로 나누기
  const mid = Math.ceil(FILTERS.length / 2);
  const topRowFilters = FILTERS.slice(0, mid);
  const bottomRowFilters = FILTERS.slice(mid);

  const renderFilterChip = (f) => (
    <button
      key={f.key}
      data-key={f.key} // CSS에서 행 맞추기 위해 data-key 추가
      className={filter === f.key ? "fav-chip fav-chip-active" : "fav-chip"}
      type="button"
      onClick={() => setFilter(f.key)}
      title={f.label}
    >
      {f.icon && (
        <span
          className="fav-chip-icon"
          style={f.iconScale ? { "--fav-icon-scale": f.iconScale } : undefined}
        >
          <img src={f.icon} alt={f.label} />
        </span>
      )}
      <span className="fav-chip-label">{f.label}</span>
    </button>
  );

  return (
    <div className="favorite-root">
      {/* 상단 바 */}
      <div className="favorite-top">
        <div className="favorite-top-inner">
          <h2 className="favorite-title">즐겨찾기</h2>

          <div className="favorite-top-right">
            <div className="fav-filter-bar">
              <div className="fav-filter-row">
                {topRowFilters.map(renderFilterChip)}
              </div>
              {bottomRowFilters.length > 0 && (
                <div className="fav-filter-row">
                  {bottomRowFilters.map(renderFilterChip)}
                </div>
              )}
            </div>

            <button
              type="button"
              className="fav-add-btn"
              onClick={openNewForm}
            >
              + 등록하기
            </button>
          </div>
        </div>
      </div>

      {/* 본문 */}
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

                  {/* 이미지 / 비디오 */}
                  <div className="fav-card-image-wrap">
                    {hasVideo ? (
                      <MediaEmbed
                        url={item.videoUrl}
                        poster={item.image}
                        className="fav-card-media"
                      />
                    ) : (
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
                    )}
                  </div>

                  {/* 텍스트 영역 */}
                  <div className="fav-card-body">
                    <div className="fav-card-body-main">
                      <div className="fav-card-text">
                        <h3
                          className="fav-card-title"
                          onClick={() => openEditForm(item)}
                        >
                          {item.title}
                        </h3>
                        <div className="fav-card-addr">
                          📍 {item.address}
                        </div>
                      </div>

                      {item.note && (
                        <p className="fav-card-note">{item.note}</p>
                      )}

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

      {/* 이미지 크롭 모달 */}
      {editingCropId && (
        <div className="fav-crop-modal-backdrop">
          <div className="fav-crop-modal">
            <div className="fav-crop-modal-header">
              <span>사진 위치 조정</span>
              <small>
                사진을 끌어서 위치를 맞추고, 확대/축소로 딱 맞게 잘라 보세요.
              </small>
            </div>

            <div className="fav-crop-frame" onMouseDown={handleCropMouseDown}>
              <img
                src={
                  favorites.find((f) => f.id === editingCropId)?.image || ""
                }
                alt="crop"
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
                  onChange={(event) =>
                    handleZoomChange(Number(event.target.value) / 100)
                  }
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
              <button
                type="button"
                className="fav-btn ghost"
                onClick={handleCropCancel}
              >
                취소
              </button>
              <button
                type="button"
                className="fav-btn ghost"
                onClick={handleCropReset}
              >
                원본으로
              </button>
              <button
                type="button"
                className="fav-btn primary"
                onClick={handleCropSave}
              >
                완료
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 등록 / 수정 폼 모달 */}
      {isFormOpen && (
        <div className="fav-form-backdrop">
          <form className="fav-form" onSubmit={handleFormSubmit}>
            <div className="fav-form-header">
              <h3 className="fav-form-title">
                {formState.id == null ? "즐겨찾기 등록" : "즐겨찾기 수정"}
              </h3>
              <p className="fav-form-subtitle">
                오늘 지나가다 본 노점, 내일 잊어버리기 전에 여기다가 한 번만
                적어두자.
              </p>
            </div>

            <div className="fav-form-section">
              {/* 사진 / 영상 업로드 */}
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

              {/* 온라인 영상 링크 */}
              <div className="fav-form-field">
                <label>온라인 영상 링크 (선택)</label>
                <input
                  type="text"
                  placeholder="예: 유튜브 / 네이버 / 카카오 등 영상 주소"
                  value={
                    formState.videoUrl.startsWith("blob:")
                      ? ""
                      : formState.videoUrl
                  }
                  onChange={(event) =>
                    handleFormChange("videoUrl", event.target.value)
                  }
                />
                <small>
                  온라인 영상 주소를 붙여넣으면 카드에서 바로 재생을 시도해요.
                  (막힌 사이트는 위 링크로 새창에서 볼 수 있음)
                </small>
              </div>

              <div className="fav-form-field">
                <label>카테고리</label>
                <select
                  value={formState.category}
                  onChange={(event) =>
                    handleFormChange("category", event.target.value)
                  }
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
                    handleFormChange(
                      "customCategoryRequest",
                      event.target.value
                    )
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
                  onChange={(event) =>
                    handleFormChange("title", event.target.value)
                  }
                  required
                />
              </div>

              <div className="fav-form-field">
                <label>위치</label>
                <input
                  type="text"
                  placeholder="예: ○○역 3번 출구 앞"
                  value={formState.address}
                  onChange={(event) =>
                    handleFormChange("address", event.target.value)
                  }
                />
              </div>

              <div className="fav-form-field">
                <label>한줄 설명</label>
                <textarea
                  rows={3}
                  placeholder="예: 줄 길지만 진짜 맛있음"
                  value={formState.note}
                  onChange={(event) =>
                    handleFormChange("note", event.target.value)
                  }
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
              <button
                type="submit"
                className="fav-btn primary"
                disabled={isSaving}
              >
                {isSaving ? "저장 중..." : "저장"}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
