// src/components/ThemeToggleButton.jsx
import React from "react";
import { useTheme } from "../theme/ThemeContext";

const ThemeToggleButton = () => {
  const { theme, toggleTheme } = useTheme();
  return (
    <button
      type="button"
      onClick={toggleTheme}
      title="테마 전환"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 10px",
        borderRadius: 999,
        border: "1px solid #e5e7eb",
        background: "#fff",
        cursor: "pointer",
        fontSize: 12,
      }}
    >
      <span aria-hidden>{theme === "dark" ? "🌙" : "☀️"}</span>
      <span>{theme === "dark" ? "라이트 모드" : "다크 모드"}</span>
    </button>
  );
};

export default ThemeToggleButton;
