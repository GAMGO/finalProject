import React from "react";
import "./Community.css";

export default function PostList({ posts }) {
  if (!posts.length) {
    return (
      <div className="post-empty">
        아직 글이 없어요.
      </div>
    );
  }

  return (
    <div className="post-list">
      {posts.map((p) => (
        <article key={p.id} className="post-card">
          <header className="post-header">
            <div className="post-header-left">
              <span className="post-type-tag">{p.type}</span>
              {p.storeCategory && (
                <span className="post-category-tag">{p.storeCategory}</span>
              )}
            </div>
            <span className="post-time">{p.createdAt}</span>
          </header>

          {p.imageUrl && (
            <div className="post-image-wrap">
              <img src={p.imageUrl} alt={p.title} />
            </div>
          )}

          <div className="post-body">
            <h3 className="post-title">{p.title}</h3>
            {p.content && <p className="post-content">{p.content}</p>}
          </div>

          <div className="post-footer">
            {p.locationText && (
              <div className="post-location">📍 {p.locationText}</div>
            )}

            <div className="post-meta-line">
              <span className="post-writer">{p.writer || "익명"}</span>
            </div>

            <div className="post-actions">
              <button className="post-icon-btn" type="button">
                ❤️ <span>{p.likeCount}</span>
              </button>
              <button className="post-icon-btn" type="button">
                💬 <span>{p.commentCount}</span>
              </button>
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}
