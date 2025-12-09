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

// ✅ 로고 이미지
import DISH_LOGO from "./assets/DISH_LOGO.png";

const Logout = ({ onLogoutSuccess }) => {
  const [message, setMessage] = useState("로그아웃을 시도 중...");

  useEffect(() => {
    const handleLogout = async () => {
      try {
        // apiClient 사용: /api/auth/logout 엔드포인트에 요청합니다.
        // apiClient의 인터셉터가 Authorization 헤더를 자동으로 추가합니다.
        await apiClient.post(`/api/auth/logout`, {});
        setMessage("로그아웃에 성공했습니다. 잠시 후 로그인 화면으로 돌아갑니다.");
        // 성공 시 onLogoutSuccess 호출 -> App 컴포넌트에서 토큰 삭제 및 isLoggedIn=false 처리
        setTimeout(() => {
          onLogoutSuccess();
        }, 1500);
      } catch (error) {
        console.error("Logout API Error:", error);
        // 서버에서 로그아웃 처리가 실패했더라도, 클라이언트 상태는 로그아웃으로 전환합니다.
        setMessage("로그아웃 처리 중 오류가 발생했지만, 인증 상태를 해제합니다.");
        setTimeout(() => {
          onLogoutSuccess();
        }, 3000);
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
  const [page, setPage] = useState("map"); // map / community / profile / favorite / logout
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // ✅ 테마 상태 (light / dark)
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "light";
    return localStorage.getItem("theme") || "light";
  });

  useEffect(() => {
    const token = localStorage.getItem("jwtToken");
    const refreshToken = localStorage.getItem("refreshToken");
    if (token) {
      setAuthToken(token, refreshToken);
      setIsLoggedIn(true);
    }
  }, []);

  // theme 값이 바뀔 때마다 <html data-theme="..."> + localStorage 동기화
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  // 사이드바에서 쓰는 토글 함수
  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  // 로그인 성공 콜백
  const handleLoginSuccess = () => {
    // 로그인/회원가입 화면에서 마지막으로 저장한 theme 값 읽어서 App state에 반영
    if (typeof window !== "undefined") {
      const savedTheme = localStorage.getItem("theme");
      if (savedTheme) {
        setTheme(savedTheme);
      }
    }

    setIsLoggedIn(true);
    setPage("map");
  };

  // 로그아웃 성공 콜백
  const handleLogoutSuccess = () => {
    clearAuthToken();
    setIsLoggedIn(false);
    setPage("map");
  };

  // 현재 활성화된 버튼 스타일 (원하면 나중에 사용)
  const getButtonClass = (targetPage) =>
    `side-nav-btn ${page === targetPage ? "active" : ""}`;

  // isLoggedIn 상태에 따른 조건부 렌더링
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
          className="side-nav-btn"
          // className={getButtonClass("community")}
          onClick={() => setPage("community")}
        >
          커뮤니티
        </button>

        <button
          className="side-nav-btn"
          // className={getButtonClass("profile")}
          onClick={() => setPage("profile")}
        >
          회원정보
        </button>

        <button
          className="side-nav-btn"
          // className={getButtonClass("favorite")}
          onClick={() => setPage("favorite")}
        >
          즐겨찾기
        </button>

        <button
          className="side-nav-btn"
          // className={getButtonClass("logout")}
          onClick={() => setPage("logout")}
        >
          로그아웃
        </button>

        {/* ✅ 로그아웃 밑에 다크/라이트 토글 버튼 */}
        <button
          type="button"
          className="side-nav-btn theme-toggle-btn"
          onClick={toggleTheme}
        >
          {theme === "light" ? "다크 모드" : "라이트 모드"}
        </button>
      </aside>

      <main className="main-content">
        {page === "map" ? (
          <Main />
        ) : page === "community" ? (
          <CommunityPage />
        ) : page === "profile" ? (
          // ✅ 수정: UserProfilePage에서 '서비스 탈퇴를 원하시나요?' 링크 클릭 시 상태를 'withdraw'로 변경
          <UserProfilePage onWithdraw={() => setPage("withdraw")} />
        ) : page === "favorite" ? (
          <FavoritePage />
        ) : page === "logout" ? (
          <Logout onLogoutSuccess={handleLogoutSuccess} />
        ) : page === "withdraw" ? ( // ✅ 핵심: 'withdraw' 상태일 때 WithdrawalPage 전체를 렌더링
          <WithdrawalPage onLogout={handleLogoutSuccess} /> 
        ) : (
          <p style={{ color: "red" }}>404 Page Not Found</p>
        )}
      </main>
    </div>
  );
}
