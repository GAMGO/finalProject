import React, { useMemo, useState, useEffect } from "react";
import "./FavoritePage.css";

// ===== 아이콘 =====
import allIcon from "../assets/favIcons/All.png";
import chickenIcon from "../assets/favIcons/chicken.png";
import ddeokbokkiIcon from "../assets/favIcons/ddeokbokki.png";
import bungebbangIcon from "../assets/favIcons/bungebbang.png";
import seafoodIcon from "../assets/favIcons/seafood.png";
import takoyakiIcon from "../assets/favIcons/takoyaki.png";
import etcIcon from "../assets/favIcons/Etc.png";

// ===== 샘플 사진 =====
import favChicken from "../assets/images/favorites/chicken.jpg";
import favBungebbang from "../assets/images/favorites/bungebbang.jpg";
import favPojangmacha from "../assets/images/favorites/pojangmacha.jpg";

const FILTERS = [
  { key: "전체", label: "전체", icon: allIcon },
  { key: "통닭", label: "통닭", icon: chickenIcon },
  { key: "분식", label: "분식", icon: ddeokbokkiIcon },
  { key: "붕어빵", label: "붕어빵", icon: bungebbangIcon },
  { key: "해산물", label: "해산물", icon: seafoodIcon },
  { key: "타코야끼", label: "타코야끼", icon: takoyakiIcon },
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

const initialFavorites = [
  {
    id: 1,
    category: "통닭",
    title: "시청 앞 통닭 트럭",
    address: "서울 중구 정동길 25 근처",
    note: "줄 길지만 진짜 맛있음",
    rating: 5.0,
    image: favChicken,
  },
  {
    id: 2,
    category: "붕어빵",
    title: "광화문 붕어빵",
    address: "광화문역 7번 출구 앞",
    note: "팥 듬뿍 + 5개 3천원",
    rating: 4.0,
    image: favBungebbang,
  },
  {
    id: 3,
    category: "분식",
    title: "을지로 떡볶이 포차",
    address: "서울 중구 어딘가",
    note: "매콤달달",
    rating: 4.5,
    image: favPojangmacha,
  },
];

export default function FavoritePage() {
  // 카테고리 필터
  const [filter, setFilter] = useState("전체");

  // 즐겨찾기 목록
  const [favorites, setFavorites] = useState(initialFavorites);

  // 이미지 초점 조절 상태
  const [cropState, setCropState] = useState(() => {
    const map = {};
    initialFavorites.forEach((fav) => {
      map[fav.id] = { ...DEFAULT_CROP };
    });
    return map;
  });

  // 크롭 모달 상태
  const [editingCropId, setEditingCropId] = useState(null);
  const [draftCrop, setDraftCrop] = useState({ ...DEFAULT_CROP });

  // 드래그 상태
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0, offsetX: 50, offsetY: 50 });

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
    customCategoryRequest: "",
  });

  // 별점 호버 상태 (.5 단위)
  const [hoverRating, setHoverRating] = useState(null);

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
  const handleCropMouseDown = (e) => {
    e.preventDefault();
    const container = e.currentTarget.getBoundingClientRect();
    setIsDragging(true);
    setDragStart({
      x: e.clientX,
      y: e.clientY,
      offsetX: draftCrop.offsetX,
      offsetY: draftCrop.offsetY,
      width: container.width,
      height: container.height,
    });
  };

  // 전역 드래그 처리
  useEffect(() => {
    if (!isDragging) return;

    const handleMove = (e) => {
      setDraftCrop((prev) => {
        const dx = e.clientX - dragStart.x;
        const dy = e.clientY - dragStart.y;

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

  // ===== 즐겨찾기 삭제 =====
  const handleUnfavorite = (id) => {
    setFavorites((prev) => prev.filter((f) => f.id !== id));
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
      customCategoryRequest: "",
    });
    setHoverRating(null);
    setIsFormOpen(true);
  };

  const handleFormChange = (field, value) => {
    setFormState((prev) => ({ ...prev, [field]: value }));
  };

  const handleFormImageChange = (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setFormState((prev) => ({ ...prev, imageUrl: url }));
  };

  // 별 클릭 시 .5 / 1.0 결정
  const getHalfOrFullValue = (starIndex, e) => {
    const native = e.nativeEvent;
    const target = native.target;
    const width = target.clientWidth || 1;
    const offsetX = native.offsetX;

    const isHalf = offsetX < width / 2;
    return isHalf ? starIndex - 0.5 : starIndex;
  };

  const handleStarClick = (starIndex, e) => {
    const value = getHalfOrFullValue(starIndex, e);
    handleFormChange("rating", value);
  };

  const handleStarHover = (starIndex, e) => {
    const value = getHalfOrFullValue(starIndex, e);
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
              onClick={(e) => handleStarClick(star, e)}
              onMouseMove={(e) => handleStarHover(star, e)}
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

  const handleFormSubmit = (e) => {
    e.preventDefault();

    const trimmedTitle = formState.title.trim();
    if (!trimmedTitle) return;

    if (formState.id == null) {
      // 새 등록
      const newId =
        favorites.length > 0
          ? Math.max(...favorites.map((f) => f.id)) + 1
          : 1;

      const newFav = {
        id: newId,
        category: formState.category,
        title: trimmedTitle,
        address: formState.address.trim(),
        note: formState.note.trim(),
        rating: Number(formState.rating) || 0,
        image:
          formState.imageUrl ||
          (formState.category === "붕어빵"
            ? favBungebbang
            : formState.category === "통닭"
            ? favChicken
            : favPojangmacha),
      };

      setFavorites((prev) => [...prev, newFav]);
      setCropState((prev) => ({
        ...prev,
        [newId]: { ...DEFAULT_CROP },
      }));
    } else {
      // 수정
      setFavorites((prev) =>
        prev.map((fav) =>
          fav.id === formState.id
            ? {
                ...fav,
                category: formState.category,
                title: trimmedTitle,
                address: formState.address.trim(),
                note: formState.note.trim(),
                rating: Number(formState.rating) || 0,
                image: formState.imageUrl || fav.image,
              }
            : fav
        )
      );
    }

    if (formState.customCategoryRequest.trim()) {
      console.log("새 카테고리 요청:", formState.customCategoryRequest.trim());
    }

    setIsFormOpen(false);
  };

  const handleFormCancel = () => {
    setIsFormOpen(false);
  };

  return (
    <div className="favorite-root">
      {/* 상단 바 */}
      <div className="favorite-top">
        <div className="favorite-top-inner">
          <h2 className="favorite-title">즐겨찾기</h2>

          <div className="favorite-top-right">
            <div className="fav-filter-bar">
              {FILTERS.map((f) => (
                <button
                  key={f.key}
                  className={
                    filter === f.key
                      ? "fav-chip fav-chip-active"
                      : "fav-chip"
                  }
                  type="button"
                  onClick={() => setFilter(f.key)}
                  title={f.label}
                >
                  {f.icon && (
                    <span className="fav-chip-icon">
                      <img src={f.icon} alt={f.label} />
                    </span>
                  )}
                  <span className="fav-chip-label">{f.label}</span>
                </button>
              ))}
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
        {filteredFavorites.length === 0 ? (
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
                        ❤️ 해제
                      </button>
                    </div>
                  </div>

                  {/* 이미지 */}
                  <div className="fav-card-image-wrap">
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
                  </div>

                  {/* 텍스트 영역 */}
                  <div className="fav-card-body">
                    <h3
                      className="fav-card-title"
                      onClick={() => openEditForm(item)}
                    >
                      {item.title}
                    </h3>
                    <div className="fav-card-addr">📍 {item.address}</div>

                    {item.note && (
                      <p className="fav-card-note">{item.note}</p>
                    )}

                    {typeof item.rating === "number" && (
                      <div className="fav-card-rating">
                        {renderStaticStars(item.rating)}
                        <span className="fav-card-rating-score">
                          {item.rating.toFixed(1)}
                        </span>
                      </div>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </div>

      {/* 이미지 크롭 모달 (1번 스샷 느낌) */}
      {editingCropId && (
        <div className="fav-crop-modal-backdrop">
          <div className="fav-crop-modal">
            <div className="fav-crop-modal-header">
              <span>사진 위치 조정</span>
              <small>사진을 끌어서 위치를 맞추고, 확대/축소로 딱 맞게 잘라 보세요.</small>
            </div>

            <div
              className="fav-crop-frame"
              onMouseDown={handleCropMouseDown}
            >
              <img
                src={
                  favorites.find((f) => f.id === editingCropId)?.image ||
                  ""
                }
                alt="crop"
                style={{
                  objectPosition: `${draftCrop.offsetX}% ${draftCrop.offsetY}%`,
                  transform: `scale(${draftCrop.zoom})`,
                }}
              />
              {/* 그리드 라인 */}
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
                  onChange={(e) => handleZoomChange(Number(e.target.value) / 100)}
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
                오늘 지나가다 본 노점, 내일 잊어버리기 전에 여기다가 한 번만 적어두자.
              </p>
            </div>

            <div className="fav-form-section">
              <div className="fav-form-field">
                <label>사진</label>
                <div className="fav-form-image-input">
                  {formState.imageUrl && (
                    <img
                      src={formState.imageUrl}
                      alt="미리보기"
                      className="fav-form-image-preview"
                    />
                  )}
                  <label className="fav-file-label">
                    파일 선택
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFormImageChange}
                      hidden
                    />
                  </label>
                </div>
              </div>

              <div className="fav-form-field">
                <label>카테고리</label>
                <select
                  value={formState.category}
                  onChange={(e) =>
                    handleFormChange("category", e.target.value)
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
                  onChange={(e) =>
                    handleFormChange("customCategoryRequest", e.target.value)
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
                  onChange={(e) => handleFormChange("title", e.target.value)}
                  required
                />
              </div>

              <div className="fav-form-field">
                <label>위치</label>
                <input
                  type="text"
                  placeholder="예: ○○역 3번 출구 앞"
                  value={formState.address}
                  onChange={(e) => handleFormChange("address", e.target.value)}
                />
              </div>

              <div className="fav-form-field">
                <label>한줄 설명</label>
                <textarea
                  rows={3}
                  placeholder="예: 줄 길지만 진짜 맛있음"
                  value={formState.note}
                  onChange={(e) => handleFormChange("note", e.target.value)}
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
              >
                취소
              </button>
              <button type="submit" className="fav-btn primary">
                저장
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
