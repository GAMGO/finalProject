// src/App.jsx
import React, { useState, useEffect } from "react";
import axios from "axios";
import Main from "./components/Main";
import CommunityPage from "./pages/CommunityPage";
import UserProfilePage from "./pages/UserProfilePage";
import WithdrawalPage from "./pages/WithdrawalPage";
import AuthPage from "./pages/AuthPage";
import FavoritePage from "./pages/FavoritePage";
import apiClient, { setAuthToken, clearAuthToken } from "./api/apiClient";
import "./App.css";
import "./SidebarPatch.css";
import "./theme/theme.css"; // ✅ 테마 CSS
import { useTheme } from "./theme/ThemeContext"; // ✅ ThemeContext 사용

// ✅ 로고 이미지
import DISH_LOGO from "./assets/DISH_LOGO.png";
// ✅ 설정 아이콘
import SETTING_ICON from "./assets/setting.png";

const Logout = ({ onLogoutSuccess }) => {
  const [message, setMessage] = useState("로그아웃을 시도 중...");

  useEffect(() => {
    const handleLogout = async () => {
      try {
        // apiClient 사용: /api/auth/logout 엔드포인트에 요청합니다. apiClient의 인터셉터가 Authorization 헤더를 자동으로 추가합니다.
        await apiClient.post(`/api/auth/logout`, {});
        setMessage("로그아웃에 성공했습니다. 잠시 후 로그인 화면으로 돌아갑니다.");
        setTimeout(() => onLogoutSuccess(), 1500);
      } catch (error) {
        console.error("Logout API Error:", error);
        setMessage("로그아웃 처리 중 오류가 발생했지만, 인증 상태를 해제합니다.");
        setTimeout(() => onLogoutSuccess(), 3000);
      }
    };
    handleLogout();
  }, [onLogoutSuccess]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        fontSize: "24px",
        fontWeight: "bold",
        color: "#333",
      }}
    >
      {message}
    </div>
  );
};

export default function App() {
  // map / community / profile / favorite / logout / withdraw
  const [page, setPage] = useState("map");
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // ✅ ThemeContext에서 전역 테마 사용
  const { theme, setTheme } = useTheme(); // ← toggleTheme 대신 setTheme 사용

  // ✅ 드롭업 열림 여부
  const [isThemeMenuOpen, setIsThemeMenuOpen] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem("jwtToken");
    const refreshToken = localStorage.getItem("refreshToken");
    if (token) {
      setAuthToken(token, refreshToken);
      setIsLoggedIn(true);
    }
  }, []);

  // ✅ 라이트/다크 직접 선택 (명시적 적용)
  const setLight = () => {
    if (theme !== "light") setTheme("light");
    setIsThemeMenuOpen(false);
  };
  const setDark = () => {
    if (theme !== "dark") setTheme("dark");
    setIsThemeMenuOpen(false);
  };

  // 로그인 성공 콜백
  const handleLoginSuccess = () => {
    setIsLoggedIn(true);
    setPage("map");
  };

  // 로그아웃 성공 콜백
  const handleLogoutSuccess = () => {
    clearAuthToken();
    setIsLoggedIn(false);
    setPage("map");
  };

  const getButtonClass = (targetPage) =>
    `side-nav-btn ${page === targetPage ? "active" : ""}`;

  // 로그인 안 했을 때는 AuthPage
  if (!isLoggedIn) {
    return (
      <div className="h-screen w-full">
        <AuthPage onLoginSuccess={handleLoginSuccess} />
      </div>
    );
  }

  return (
    <div className="app">
      <aside className="side-bar">
        {/* ✅ 왼쪽 상단 로고 */}
        <div className="side-logo" onClick={() => setPage("map")}>
          <img
            src={DISH_LOGO}
            alt="DISHINSIDE 로고"
            className="side-logo-img"
          />
        </div>

        <button
          className={getButtonClass("community")}
          onClick={() => setPage("community")}
        >
          커뮤니티
        </button>

        <button
          className={getButtonClass("profile")}
          onClick={() => setPage("profile")}
        >
          회원정보
        </button>

        <button
          className={getButtonClass("favorite")}
          onClick={() => setPage("favorite")}
        >
          즐겨찾기
        </button>

        {/* ✅ 왼쪽 하단 설정 아이콘 + 작은 로그아웃 버튼 + 드롭업 메뉴 */}
        <div className="side-settings-wrap">
          <button
            type="button"
            className="side-settings-btn"
            onClick={() => setIsThemeMenuOpen((prev) => !prev)}
            title="테마 설정"
          >
            <img
              src={SETTING_ICON}
              alt="테마 설정"
              className="side-settings-icon"
            />
          </button>

          {/* 🔓 오른쪽에 작은 로그아웃 버튼 */}
          <button
            type="button"
            className="side-logout-mini"
            onClick={() => setPage("logout")}
          >
            로그아웃
          </button>

          {isThemeMenuOpen && (
            <div className="side-settings-menu">
              <button
                type="button"
                className={
                  "side-settings-item" + (theme === "light" ? " active" : "")
                }
                onClick={setLight}
              >
                라이트 모드
              </button>
              <button
                type="button"
                className={
                  "side-settings-item" + (theme === "dark" ? " active" : "")
                }
                onClick={setDark}
              >
                다크 모드
              </button>
            </div>
          )}
        </div>
      </aside>

      <main className="main-content">
        {page === "map" ? (
          <Main />
        ) : page === "community" ? (
          <CommunityPage />
        ) : page === "profile" ? (
          <UserProfilePage onWithdraw={() => setPage("withdraw")} />
        ) : page === "favorite" ? (
          <FavoritePage />
        ) : page === "logout" ? (
          <Logout onLogoutSuccess={handleLogoutSuccess} />
        ) : page === "withdraw" ? (
          <WithdrawalPage onLogout={handleLogoutSuccess} />
        ) : (
          <p style={{ color: "red" }}>404 Page Not Found</p>
        )}
      </main>
    </div>
  );
}
