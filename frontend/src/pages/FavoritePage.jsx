import React, { useMemo, useState } from "react";
import "./FavoritePage.css";

const FILTERS = [
  { key: "전체", label: "전체", icon: "⭐" },
  { key: "통닭", label: "통닭", icon: "🍗" },
  { key: "타코야끼", label: "타코야끼", icon: "🐙" },
  { key: "붕어빵", label: "붕어빵", icon: "🐟" },
  { key: "분식", label: "분식", icon: "🍢" },
  { key: "해산물", label: "해산물", icon: "🦐" },
  { key: "기타", label: "기타", icon: "⋯" },
];

// 더미. 나중에 API 붙이면 이 배열만 교체하면 됨.
const dummyFavorites = [
  {
    id: 1,
    category: "통닭",
    title: "시청 앞 통닭 트럭",
    address: "서울 중구 정동길 25 근처",
    note: "줄 길지만 진짜 맛있음",
  },
  {
    id: 2,
    category: "붕어빵",
    title: "광화문 붕어빵",
    address: "광화문역 7번 출구 앞",
    note: "팥 듬뿍 + 5개 3천원",
  },
];

export default function FavoritePage() {
  const [filter, setFilter] = useState("전체");
  const favorites = dummyFavorites;

  const filtered = useMemo(() => {
    if (filter === "전체") return favorites;
    return favorites.filter((f) => f.category === filter);
  }, [filter, favorites]);

  return (
    <div className="favorite-root">
      {/* 상단 보라색 헤더 (커뮤니티랑 같은 톤) */}
      <div className="favorite-top">
        <div className="favorite-top-inner">
          <h2 className="favorite-title">즐겨찾기</h2>

          {/* 즐겨찾기 전용 필터바: 원형 재탕 X */}
          <div className="fav-filter-bar">
            {FILTERS.map((f) => (
              <button
                key={f.key}
                className={`fav-chip ${filter === f.key ? "fav-chip-active" : ""}`}
                onClick={() => setFilter(f.key)}
                type="button"
              >
                <span className="fav-chip-icon" aria-hidden>
                  {f.icon}
                </span>
                <span className="fav-chip-label">{f.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 리스트 */}
      <div className="favorite-inner">
        {!filtered.length ? (
          <div className="fav-empty">즐겨찾기한 노점이 없어요.</div>
        ) : (
          <div className="fav-list">
            {filtered.map((item) => (
              <article key={item.id} className="fav-card">
                <div className="fav-card-badge">{item.category}</div>
                <div className="fav-card-right">
                  <button className="fav-unlike" type="button">
                    ❤️ 해제
                  </button>
                </div>

                <h3 className="fav-card-title">{item.title}</h3>
                <div className="fav-card-addr">📍 {item.address}</div>
                {item.note && <p className="fav-card-note">{item.note}</p>}
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
