// src/App.jsx
import React, { useState, useEffect } from "react";
import axios from "axios";
import Main from "./components/Main";
import CommunityPage from "./pages/CommunityPage";
import UserProfilePage from "./pages/UserProfilePage";
import AuthPage from "./pages/AuthPage"
import FavoritePage from "./pages/FavoritePage";
import apiClient, { setAuthToken, clearAuthToken } from "./api/apiClient";
import "./App.css";
import "./SidebarPatch.css";

// ✅ 로고 이미지
import DISH_LOGO from "./assets/DISH_LOGO.png";

const Logout = ({ onLogoutSuccess }) => {
    const [message, setMessage] = useState("로그아웃을 시도 중...");
    useEffect(() => {
        const handleLogout = async () => {
            try {
                //apiClient 사용: /api/auth/logout 엔드포인트에 요청합니다. apiClient의 인터셉터가 Authorization 헤더를 자동으로 추가합니다.
                await apiClient.post(
                    `/api/auth/logout`,
                    {}
                );
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
        <AuthPage onLoginSuccess={handleLoginSuccess}/>
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
          <Main />
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
