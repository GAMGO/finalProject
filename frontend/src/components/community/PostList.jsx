// src/components/community/PostList.jsx
import React from "react";
import "./Community.css";

export default function PostList({ posts }) {
  if (!posts || posts.length === 0) {
    return <div className="post-empty">아직 올라온 글이 없어요.</div>;
  }

  return (
    <ul className="post-list">
      {posts.map((post) => (
        <li key={post.id} className="post-row">
          {/* 왼쪽 작은 이미지 */}
          <div className="post-thumb">
            {post.imageUrl ? (
              <img src={post.imageUrl} alt={post.title} />
            ) : (
              <div className="post-thumb-placeholder" />
            )}
          </div>

          {/* 가운데 제목 + 위치 + 시간 */}
          <div className="post-row-main">
            <div className="post-row-title-line">
              {post.type && (
                <span className="post-row-type">{post.type}</span>
              )}
              <span className="post-row-title">{post.title}</span>
            </div>
            <div className="post-row-meta">
              {post.locationText && (
                <>
                  <span className="post-row-location">
                    {post.locationText}
                  </span>
                  <span className="post-row-dot">·</span>
                </>
              )}
              <span className="post-row-time">{post.createdAt}</span>
              {post.writer && (
                <>
                  <span className="post-row-dot">·</span>
                  <span className="post-row-writer">{post.writer}</span>
                </>
              )}
            </div>
          </div>

          {/* 오른쪽 좋아요 / 싫어요 */}
          <div className="post-row-actions">
            <button className="post-row-action-btn">
              👍 <span>{post.likeCount ?? 0}</span>
            </button>
            <button className="post-row-action-btn">
              👎 <span>{post.dislikeCount ?? 0}</span>
            </button>
          </div>
        </li>
      ))}
    </ul>
  );
}
