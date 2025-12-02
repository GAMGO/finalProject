// src/App.jsx
import React, { useState, useEffect } from "react";
import axios from "axios";
import KakaoMap from "./components/KakaoMap";
import CommunityPage from "./pages/CommunityPage";
import UserProfilePage from "./pages/UserProfilePage";
import FavoritePage from "./pages/FavoritePage";
import apiClient, { setAuthToken, clearAuthToken } from "./api/apiClient";
import "./App.css";
import "./SidebarPatch.css";

// ✅ 로고 이미지
import DISH_LOGO from "./assets/DISH_LOGO.png";

const API_BASE_URL = "http://localhost:8080";

// ⭐️ 로그인 화면 컴포넌트 (로그아웃 시 전체 화면 대체)
const LoginPage = ({ onLoginSuccess }) => (
  <div
    className="login-page-wrapper"
    style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      height: "100vh",
      width: "100%",
      backgroundColor: "#f0f0f0",
    }}
  >
    <div
      className="login-card"
      style={{
        backgroundColor: "white",
        padding: "40px",
        borderRadius: "12px",
        boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
        textAlign: "center",
        width: "350px",
      }}
    >
      <h1
        style={{
          fontSize: "28px",
          fontWeight: "bold",
          marginBottom: "20px",
          color: "#3f51b5",
        }}
      >
        로그인 필요
      </h1>
      <p style={{ marginBottom: "30px", color: "#555" }}>
        로그아웃 상태입니다. 서비스를 이용하려면 로그인해 주세요.
      </p>
      <button
        onClick={() => {
          // ⭐️ 실제 로그인 API 호출 후 토큰을 받아서 onLoginSuccess(token)을 호출해야 합니다.
          // 현재는 데모를 위해 임시 토큰을 사용합니다.
          const mockToken =
            "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ6eGMxMjMiLCJpc3MiOiJjb20uZXhhbXBsZSIsImlhdCI6MTc2NDU2NDQ5MywiZXhwIjoxNzY0NjUwODkzfQ.JRzFU54tzCAqHIrvBdcASsuI8-1a2JlfHFioiXv9Aofp5g9y1vYKwLMe66ZE2ni9xfQONktSyg13rEMpDWrjMg";
          onLoginSuccess(mockToken);
        }}
        style={{
          width: "100%",
          backgroundColor: "#3f51b5",
          color: "white",
          fontWeight: "bold",
          padding: "12px",
          borderRadius: "8px",
          border: "none",
          cursor: "pointer",
          transition: "background-color 0.2s",
        }}
      >
        로그인하기 (클릭하여 앱 재시작)
      </button>
    </div>
  </div>
);

const Logout = ({ onLogoutSuccess }) => {
  const [message, setMessage] = useState("로그아웃을 시도 중...");

  useEffect(() => {
    const handleLogout = async () => {
      try {
        await apiClient.post(`/api/auth/logout`, {});
        setMessage(
          "로그아웃에 성공했습니다. 잠시 후 로그인 화면으로 돌아갑니다."
        );
        setTimeout(() => {
          onLogoutSuccess();
        }, 1500);
      } catch (error) {
        console.error("Logout API Error:", error);
        setMessage(
          "로그아웃 처리 중 오류가 발생했지만, 인증 상태를 해제합니다."
        );
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

  // 로그인 성공 콜백: 토큰을 받아 메모리에 저장합니다.
  const handleLoginSuccess = (token) => {
    setAuthToken(token);
    setIsLoggedIn(true);
    setPage("map");
  };

  // 로그아웃 성공 콜백
  const handleLogoutSuccess = () => {
    clearAuthToken();
    setIsLoggedIn(false);
  };

  // 현재 활성화된 버튼 스타일 (원하면 나중에 사용)
  const getButtonClass = (targetPage) =>
    `side-nav-btn ${page === targetPage ? "active" : ""}`;

  // isLoggedIn 상태에 따른 조건부 렌더링
  if (!isLoggedIn) {
    return (
      <div className="h-screen w-full">
        <LoginPage onLoginSuccess={handleLoginSuccess} />
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
      </aside>

      <main className="main-content">
        {page === "map" ? (
          <KakaoMap />
        ) : page === "community" ? (
          <CommunityPage />
        ) : page === "profile" ? (
          <UserProfilePage />
        ) : page === "favorite" ? (
          <FavoritePage />
        ) : page === "logout" ? (
          <Logout onLogoutSuccess={handleLogoutSuccess} />
        ) : (
          <p style={{ color: "red" }}>404 Page Not Found</p>
        )}
      </main>
    </div>
  );
}
