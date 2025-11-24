import React, { useState } from "react";
import "./Community.css";

const TYPE_OPTIONS = ["제보", "후기", "질문", "잡담"];

export default function PostEditor({ onClose, onSubmit }) {
  const [type, setType] = useState("제보");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [storeCategory, setStoreCategory] = useState("");
  const [locationText, setLocationText] = useState("");
  const [writer, setWriter] = useState("");
  const [imagePreview, setImagePreview] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImagePreview(url);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title.trim()) return;

    onSubmit({
      type,
      title,
      content,
      storeCategory,
      locationText,
      writer: writer || "익명",
      imageUrl: imagePreview,
    });
  };

  return (
    <div className="editor-backdrop">
      <div className="editor-panel">
        <div className="editor-header">
          <h3>글쓰기</h3>
          <button className="editor-close" type="button" onClick={onClose}>
            ✕
          </button>
        </div>

        <form className="editor-form" onSubmit={handleSubmit}>
          <div className="editor-row">
            <label>종류</label>
            <select value={type} onChange={(e) => setType(e.target.value)}>
              {TYPE_OPTIONS.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          <div className="editor-row">
            <label>제목</label>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="제목을 입력하세요"
            />
          </div>

          <div className="editor-row">
            <label>내용</label>
            <textarea
              rows={4}
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="오늘 본 노점, 후기, 팁 등을 적어주세요."
            />
          </div>

          <div className="editor-row">
            <label>위치</label>
            <input
              value={locationText}
              onChange={(e) => setLocationText(e.target.value)}
              placeholder="예: 서울 중구 정동길 25 근처"
            />
          </div>

          <div className="editor-row">
            <label>카테고리</label>
            <input
              value={storeCategory}
              onChange={(e) => setStoreCategory(e.target.value)}
              placeholder="예: 붕어빵, 통닭, 토스트…"
            />
          </div>

          <div className="editor-row">
            <label>작성자</label>
            <input
              value={writer}
              onChange={(e) => setWriter(e.target.value)}
              placeholder="닉네임 (안 쓰면 익명)"
            />
          </div>

          <div className="editor-row">
            <label>사진</label>
            <div className="editor-image-field">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
              />
              {imagePreview && (
                <div className="editor-image-preview">
                  <img src={imagePreview} alt="preview" />
                </div>
              )}
            </div>
          </div>

          <div className="editor-actions">
            <button type="button" onClick={onClose}>
              취소
            </button>
            <button type="submit" className="editor-submit">
              게시하기
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
