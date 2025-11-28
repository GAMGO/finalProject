// src/components/community/PostCard.jsx
import React from "react";

export default function PostCard({ post }) {
  return (
    <article className="comm-card">
      <div className="comm-card-top">
        <span className={`comm-pill comm-pill--type`}>
          {post.type}
        </span>
        {post.storeCategory && (
          <span className="comm-pill comm-pill--category">
            {post.storeCategory}
          </span>
        )}
      </div>

      <h3 className="comm-card-title">{post.title}</h3>

      {post.content && (
        <p className="comm-card-content">{post.content}</p>
      )}

      {post.locationText && (
        <div className="comm-card-location">
          📍 {post.locationText}
        </div>
      )}

      <div className="comm-card-meta">
        <span className="comm-meta-left">
          {post.writer} · {post.createdAt}
        </span>
        <span className="comm-meta-right">
          ❤️ {post.likeCount} · 💬 {post.commentCount}
        </span>
      </div>
    </article>
  );
}
