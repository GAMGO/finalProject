// src/components/NewSessionModal.jsx
import { useEffect, useState } from "react";
import "../styles/NewSessionModal.css";

export default function NewSessionModal({ open, onClose, onConfirm }) {
  const [title, setTitle] = useState("");

  // 모달 열릴 때마다 입력값 초기화 + 자동 포커스
  useEffect(() => {
    if (open) {
      setTitle("");
      setTimeout(() => {
        const el = document.getElementById("new-session-input");
        if (el) el.focus();
      }, 0);
    }
  }, [open]);

  if (!open) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = title.trim();
    if (!trimmed) return;
    onConfirm(trimmed);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-card"
        onClick={(e) => e.stopPropagation()} // 바깥 클릭하면만 닫힘
      >
        <div className="modal-header">
          <div className="modal-title">새 대화 제목</div>
          <button
            className="modal-close"
            type="button"
            onClick={onClose}
            aria-label="닫기"
          >
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <label className="modal-label">
            <span className="modal-label-text">대화 제목</span>
            <input
              id="new-session-input"
              type="text"
              className="modal-input"
              placeholder="예: 출장비 정리용 엑셀 양식"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
          </label>

          <div className="modal-footer">
            <button
              type="button"
              className="modal-btn secondary"
              onClick={onClose}
            >
              취소
            </button>
            <button
              type="submit"
              className="modal-btn primary"
              disabled={!title.trim()}
            >
              대화 만들기
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
