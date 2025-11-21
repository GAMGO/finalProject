// src/components/PlaceCard.jsx
import React from 'react';

export function PlaceCard({ place, onToggleFavorite, onLike }) {
  const {
    status, // "DISCOVERED" or "VERIFIED"
    name,
    category,
    address,
    mainMenu,
    likeCount,
    avgRating,
    representativeReview,
    isFavorite,
  } = place;

  const statusLabel = status === 'VERIFIED' ? '인증됨' : '발굴됨';

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <span style={styles.badge}>{statusLabel}</span>
        {name && <h3 style={{ margin: 0 }}>{name}</h3>}
      </div>
      <p>카테고리: {category}</p>
      <p>주소: {address}</p>
      {mainMenu && <p>대표 메뉴: {mainMenu}</p>}

      <div style={styles.metaRow}>
        <button onClick={onToggleFavorite} style={styles.iconButton}>
          {isFavorite ? '❤️' : '🤍'}
        </button>
        <button onClick={onLike} style={styles.iconButton}>
          👍 {likeCount}
        </button>
        <span>⭐ {avgRating?.toFixed ? avgRating.toFixed(1) : avgRating}</span>
      </div>

      {representativeReview && (
        <div style={styles.reviewBox}>
          <strong>대표 리뷰</strong>
          <p>{representativeReview.content}</p>
          <small>좋아요 {representativeReview.likeCount}</small>
          {representativeReview.owner && (
            <span style={styles.ownerBadge}>사장</span>
          )}
        </div>
      )}

      {/* 여기에는 카카오맵 영역을 나중에 추가 */}
      <div style={styles.mapPlaceholder}>[지도 자리 - Kakao Map]</div>
    </div>
  );
}

const styles = {
  card: {
    border: '1px solid #ddd',
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
    width: 360,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  badge: {
    fontSize: 12,
    padding: '2px 6px',
    borderRadius: 4,
    backgroundColor: '#eee',
  },
  metaRow: {
    display: 'flex',
    gap: 8,
    alignItems: 'center',
    marginTop: 8,
    marginBottom: 8,
  },
  iconButton: {
    border: 'none',
    background: 'none',
    cursor: 'pointer',
    fontSize: 16,
  },
  reviewBox: {
    borderTop: '1px solid #eee',
    paddingTop: 8,
    marginTop: 8,
  },
  ownerBadge: {
    marginLeft: 8,
    fontSize: 10,
    padding: '1px 4px',
    borderRadius: 4,
    backgroundColor: '#ffe58f',
  },
  mapPlaceholder: {
    marginTop: 8,
    height: 120,
    borderRadius: 8,
    backgroundColor: '#f5f5f5',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 12,
    color: '#888',
  },
};

/*
 * [파일 설명]
 * - 가게 한 개를 카드 형식으로 보여주는 UI 컴포넌트.
 * - 발굴됨/인증됨, 좋아요, 즐겨찾기, 대표리뷰, 지도 자리 등
 *   네가 요구한 정보 구조대로 배치해 둔 기본 틀.
 * - 실제 onToggleFavorite/onLike에서 /favorite, /like API 호출 붙이면 됨.
 */
