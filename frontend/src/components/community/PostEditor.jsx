// src/components/community/PostEditor.jsx
import React, { useState, useEffect } from "react";
import "./Community.css";

export default function PostEditor({ onClose, onSubmit, initial }) {
  const [form, setForm] = useState({
    type: "제보",
    title: "",
    body: "",
    locationText: "",
    storeCategory: "",
    writer: "",
    imageUrl: "",
  });

  // 초기값 반영 (수정 모드일 때)
  useEffect(() => {
    if (initial) {
      setForm((prev) => ({ ...prev, ...initial }));
    }
  }, [initial]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setForm((prev) => ({ ...prev, imageUrl: url, file }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      type: form.type,
      title: form.title,
      body: form.body,
      locationText: form.locationText,
      storeCategory: form.storeCategory,
      writer: form.writer,
      imageUrl: form.imageUrl,
      file: form.file, // (선택) 파일 업로드 분리형이면 services에서 처리
    });
  };

  const isEdit = Boolean(initial);

  return (
    <div className="post-editor-root">
      <div className="post-editor-card">
        <div className="post-editor-header">
          <h2>{isEdit ? "글 수정" : "글쓰기"}</h2>
          <button type="button" className="post-editor-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <form className="post-editor-form" onSubmit={handleSubmit}>
          <div className="post-editor-row">
            <label>종류</label>
            <select
              name="type"
              value={form.type}
              onChange={handleChange}
              className="post-editor-input"
            >
              <option value="제보">제보</option>
              <option value="후기">후기</option>
              <option value="질문">질문</option>
              <option value="잡담">잡담</option>
            </select>
          </div>

          <div className="post-editor-row">
            <label>제목</label>
            <input
              name="title"
              value={form.title}
              onChange={handleChange}
              placeholder="제목을 입력하세요"
              className="post-editor-input"
              required
            />
          </div>

          <div className="post-editor-row">
            <label>위치</label>
            <input
              name="locationText"
              value={form.locationText}
              onChange={handleChange}
              placeholder="예: 서울 중구 정동길 25 근처"
              className="post-editor-input"
            />
          </div>

          <div className="post-editor-row">
            <label>카테고리</label>
            <input
              name="storeCategory"
              value={form.storeCategory}
              onChange={handleChange}
              placeholder="예: 매운탕, 통닭, 토스트…"
              className="post-editor-input"
            />
          </div>

          <div className="post-editor-row">
            <label>작성자</label>
            <input
              name="writer"
              value={form.writer}
              onChange={handleChange}
              placeholder="닉네임 (안 쓰면 익명)"
              className="post-editor-input"
            />
          </div>

          <div className="post-editor-row post-editor-row-textarea">
            <label>내용</label>
            <textarea
              name="body"
              value={form.body}
              onChange={handleChange}
              rows={6}
              placeholder="오늘 본 노점, 후기, TIP 등을 적어주세요."
              className="post-editor-textarea"
              required
            />
          </div>

          <div className="post-editor-row post-editor-row-file">
            <label>사진</label>
            <div className="post-editor-file-wrap">
              <input type="file" accept="image/*" onChange={handleFileChange} />
              {form.imageUrl && (
                <div className="post-editor-thumb">
                  <img src={form.imageUrl} alt="미리보기" />
                </div>
              )}
            </div>
          </div>

          <div className="post-editor-actions">
            <button
              type="button"
              className="post-editor-btn post-editor-btn-secondary"
              onClick={onClose}
            >
              취소
            </button>
            <button type="submit" className="post-editor-btn post-editor-btn-primary">
              {isEdit ? "수정하기" : "게시하기"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
