// src/theme/ThemeContext.jsx
import React, { createContext, useContext, useEffect, useState } from "react";

// light | dark
const ThemeContext = createContext({
  theme: "light",
  setTheme: () => {},
  toggleTheme: () => {},
});

export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "light";
    return localStorage.getItem("dishinside-theme") || "light";
  });

  useEffect(() => {
    const root = document.documentElement;

    // ✅ data-theme 속성으로 통일 (CSS와 일치)
    root.setAttribute("data-theme", theme);

    // ✅ classList는 필요 시 유지 가능 (선택사항)
    root.classList.remove("theme-light", "theme-dark");
    root.classList.add(theme === "dark" ? "theme-dark" : "theme-light");

    localStorage.setItem("dishinside-theme", theme);
  }, [theme]);

  // ✅ 명시적 setTheme 외에 toggleTheme도 그대로 제공
  const toggleTheme = () =>
    setTheme((prev) => (prev === "light" ? "dark" : "light"));

  const value = { theme, setTheme, toggleTheme };

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);
