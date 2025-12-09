// src/components/ThemeToggleButton.jsx
import React from "react";
import { useTheme } from "../theme/ThemeContext";

const ThemeToggleButton = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <button className="theme-toggle-btn" onClick={toggleTheme}>
      {theme === "light" ? "다크 모드" : "라이트 모드"}
    </button>
  );
};

export default ThemeToggleButton;
