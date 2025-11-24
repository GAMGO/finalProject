import React from "react";
import "./Community.css";

const FILTERS = ["전체", "제보", "후기", "질문", "잡담"];

export default function PostFilterBar({ current, onChange, onClickWrite }) {
  return (
    <div className="filter-bar">
      <div className="story-strip">
        {FILTERS.map((label) => (
          <button
            key={label}
            className={
              "story-chip" + (current === label ? " story-chip-active" : "")
            }
            onClick={() => onChange(label)}
          >
            <div className="story-chip-circle" />
            <span className="story-chip-label">{label}</span>
          </button>
        ))}
      </div>

      <button className="write-big-btn" onClick={onClickWrite}>
        + 글쓰기
      </button>
    </div>
  );
}
