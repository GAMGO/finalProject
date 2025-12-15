// src/pages/FavoritePage.jsx
import React, { useMemo, useState, useEffect } from "react";
import "./FavoritePage.css";
import { favoriteApi } from "../api/apiClient";

/** 폼에서 카테고리 선택용 */
const FILTERS = [
  { key: "통닭", label: "통닭" },
  { key: "타코야끼", label: "타코야끼" },
  { key: "순대곱창", label: "순대·곱창" },
  { key: "붕어빵", label: "붕어빵" },
  { key: "군밤/고구마", label: "군밤/고구마" },
  { key: "닭꼬치", label: "닭꼬치" },
  { key: "분식", label: "분식" },
  { key: "해산물", label: "해산물" },
  { key: "뻥튀기", label: "뻥튀기" },
  { key: "계란빵", label: "계란빵" },
  { key: "옥수수", label: "옥수수" },
  { key: "기타", label: "기타" },
];

const CATEGORY_ALIAS = {
  분식: ["분식", "떡볶이"],
};

// ✅ 숫자 통일(혹시 문자열로 오는 경우 대비)
const toNum = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

const mapFromDto = (dto) => {
  const id = dto.id ?? dto.idx ?? dto.IDX;

  // ✅ 찜 연결 핵심: favoriteStoreIdx도 같이 들고 다니기
  const favoriteStoreIdx =
    dto.favoriteStoreIdx ??
    dto.favorite_store_idx ??
    dto.FAVORITE_STORE_IDX ??
    null;

  const category = dto.category ?? dto.CATEGORY ?? "기타";
  const title = dto.title ?? dto.TITLE ?? "";
  const favoriteAddress =
    dto.favoriteAddress ?? dto.FAVORITE_ADDRESS ?? dto.address ?? "";

  const note = dto.note ?? dto.NOTE ?? "";
  const ratingRaw = dto.rating ?? dto.RATING ?? 0;
  const rating =
    typeof ratingRaw === "number" ? ratingRaw : Number(ratingRaw) || 0;

  // 사진/영상 필드는 데이터로는 유지(수정폼에서 쓸 수도 있어서)
  const imageUrl = dto.imageUrl ?? dto.IMAGE_URL ?? "";
  const videoUrl = dto.videoUrl ?? dto.VIDEO_URL ?? "";

  return {
    id: toNum(id),
    favoriteStoreIdx: toNum(favoriteStoreIdx),
    category,
    title,
    address: favoriteAddress,
    note,
    rating,
    image: imageUrl || "",
    videoUrl: videoUrl || "",
    createdAt: dto.createdAt ?? dto.CREATED_AT ?? null,
    expiredAt: dto.expiredAt ?? dto.EXPIRED_AT ?? null,
  };
};

export default function FavoritePage({ categoryFilter = "전체" }) {
  // ✅ 사이드바에서 내려온 값만 사용
  const filter = categoryFilter || "전체";

  const [favorites, setFavorites] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const [isFormOpen, setIsFormOpen] = useState(false);

  // ✅ favoriteStoreIdx 포함 (수정 저장 시 null로 덮어쓰기 방지)
  const [formState, setFormState] = useState({
    id: null,
    favoriteStoreIdx: null,
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
      setFavorites([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchFavorites();
  }, []);

  const filteredFavorites = useMemo(() => {
    if (filter === "전체") return favorites;

    if (CATEGORY_ALIAS[filter]) {
      const aliases = CATEGORY_ALIAS[filter];
      return favorites.filter((f) => aliases.includes(f.category));
    }

    return favorites.filter((f) => f.category === filter);
  }, [favorites, filter]);

  const handleUnfavorite = async (id) => {
    if (!window.confirm("이 즐겨찾기를 해제할까요?")) return;

    try {
      await favoriteApi.remove(id);
      setFavorites((prev) => prev.filter((f) => f.id !== id));
    } catch (error) {
      console.error("즐겨찾기 해제 실패", error);
      alert("즐겨찾기 해제에 실패했습니다.");
    }
  };

  const openEditForm = (fav) => {
    setFormState({
      id: fav.id,
      favoriteStoreIdx: fav.favoriteStoreIdx ?? null,
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

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    if (isSaving) return;

    const trimmedTitle = formState.title.trim();
    if (!trimmedTitle) return;

    const trimmedVideoUrl =
      typeof formState.videoUrl === "string"
        ? formState.videoUrl.trim()
        : formState.videoUrl || "";

    const stableFavoriteStoreIdx =
      formState.favoriteStoreIdx ??
      favorites.find((f) => f.id === formState.id)?.favoriteStoreIdx ??
      null;

    const payload = {
      idx: formState.id ?? null,
      favoriteStoreIdx: stableFavoriteStoreIdx,
      category: formState.category,
      title: trimmedTitle,
      favoriteAddress: formState.address.trim(),
      note: formState.note.trim(),
      rating: Number(formState.rating) || 0,
      imageUrl: formState.imageUrl || "",
      videoUrl: trimmedVideoUrl || "",
    };

    try {
      setIsSaving(true);
      const updatedDto = await favoriteApi.update(formState.id, payload);
      const updated = mapFromDto(updatedDto);

      setFavorites((prev) =>
        prev.map((fav) => (fav.id === updated.id ? updated : fav))
      );

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
      <div className="favorite-inner">
        {isLoading ? (
          <div className="fav-empty">즐겨찾기를 불러오는 중입니다...</div>
        ) : filteredFavorites.length === 0 ? (
          <div className="fav-empty">즐겨찾기한 노점이 없어요.</div>
        ) : (
          <div className="fav-list">
            {filteredFavorites.map((item) => {
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

                  {/* ✅ 사진/영상 영역 완전 제거 */}

                  <div className="fav-card-body">
                    <div className="fav-card-body-main">
                      <div className="fav-card-text">
                        <h3
                          className="fav-card-title"
                          onClick={() => openEditForm(item)}
                        >
                          {item.title}
                        </h3>
                        <div className="fav-card-addr">📍 {item.address}</div>
                      </div>

                      {typeof item.rating === "number" && (
                        <div className="fav-card-rating fav-card-rating-right">
                          {renderStaticStars(item.rating)}
                          <span className="fav-card-rating-score">
                            {item.rating.toFixed(1)}
                          </span>
                        </div>
                      )}

                      {item.note && <p className="fav-card-note">{item.note}</p>}

                      {/* (선택) 영상 링크만 텍스트로 남기고 싶으면 이거 켜 */}
                      {/* {item.videoUrl && (
                        <a className="fav-card-link" href={item.videoUrl} target="_blank" rel="noreferrer">
                          영상 링크 열기
                        </a>
                      )} */}
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </div>

      {/* ===== 수정 폼 모달 ===== */}
      {isFormOpen && (
        <div className="fav-form-backdrop">
          <form className="fav-form" onSubmit={handleFormSubmit}>
            <div className="fav-form-header">
              <h3 className="fav-form-title">즐겨찾기 수정</h3>
              <p className="fav-form-subtitle">
                지나가다 본 노점, 기억날 때 후딱 수정해두자.
              </p>
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
                <small>카드에는 사진을 표시하지 않지만, 데이터로는 저장/수정 가능.</small>
              </div>

              <div className="fav-form-field">
                <label>온라인 영상 링크 (선택)</label>
                <input
                  type="text"
                  placeholder="예: 유튜브 / 네이버 / 카카오 등 영상 주소"
                  value={formState.videoUrl.startsWith("blob:") ? "" : formState.videoUrl}
                  onChange={(event) =>
                    handleFormChange("videoUrl", event.target.value)
                  }
                />
              </div>

              <div className="fav-form-field">
                <label>카테고리</label>
                <select
                  value={formState.category}
                  onChange={(event) =>
                    handleFormChange("category", event.target.value)
                  }
                >
                  {FILTERS.map((f) => (
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
