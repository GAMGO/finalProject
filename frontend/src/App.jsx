import React, { useState, useEffect } from "react";
import KakaoMap from "./components/KakaoMap";
import CommunityPage from "./pages/CommunityPage";
import UserProfilePage from "./pages/UserProfilePage";
import WithdrawalPage from "./pages/WithdrawalPage";
import AuthPage from "./pages/AuthPage";
import FavoritePage from "./pages/FavoritePage";
import apiClient, { setAuthToken, clearAuthToken } from "./api/apiClient";

import "./App.css";
import "./SidebarPatch.css";
import "./theme/theme.css";
import { useTheme } from "./theme/ThemeContext";

import DISH_LOGO from "./assets/DISH_LOGO.png";
import SETTING_ICON from "./assets/setting.png";

import { CATEGORIES } from "./constants/categories";

const Logout = ({ onLogoutSuccess }) => {
  const [message, setMessage] = useState("로그아웃을 시도 중...");

  useEffect(() => {
    const handleLogout = async () => {
      try {
        await apiClient.post(`/api/auth/logout`, {});
        setMessage("로그아웃에 성공했습니다. 잠시 후 로그인 화면으로 돌아갑니다.");
        setTimeout(() => onLogoutSuccess(), 1500);
      } catch (error) {
        setMessage("로그아웃 처리 중 오류가 발생했지만, 인증 상태를 해제합니다.");
        setTimeout(() => onLogoutSuccess(), 1500);
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

// ✅ 즐겨찾기 페이지용(문자열 필터)
const FAVORITE_CATEGORY_OPTIONS = [
  { value: "전체", label: "전체" },
  { value: "통닭", label: "통닭" },
  { value: "타코야끼", label: "타코야끼" },
  { value: "순대곱창", label: "순대·곱창" },
  { value: "붕어빵", label: "붕어빵" },
  { value: "군밤/고구마", label: "군밤/고구마" },
  { value: "닭꼬치", label: "닭꼬치" },
  { value: "분식", label: "분식" },
  { value: "해산물", label: "해산물" },
  { value: "뻥튀기", label: "뻥튀기" },
  { value: "계란빵", label: "계란빵" },
  { value: "옥수수", label: "옥수수" },
  { value: "기타", label: "기타" },
];

export default function App() {
  const [page, setPage] = useState("map");
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const { theme, setTheme } = useTheme();
  const [isThemeMenuOpen, setIsThemeMenuOpen] = useState(false);

  // ✅ MAP 페이지: 카테고리 필터(문자열 id)
  const [categoryFilterId, setCategoryFilterId] = useState("");

  // ✅ FAVORITE 페이지: 카테고리 필터(문자열 key)
  const [favoriteCategoryFilter, setFavoriteCategoryFilter] = useState("전체");

  useEffect(() => {
    const token = localStorage.getItem("jwtToken");
    const refreshToken = localStorage.getItem("refreshToken");
    if (token) {
      setAuthToken(token, refreshToken);
      setIsLoggedIn(true);
    }
  }, []);

  const setLight = () => {
    if (theme !== "light") setTheme("light");
    setIsThemeMenuOpen(false);
  };
  const setDark = () => {
    if (theme !== "dark") setTheme("dark");
    setIsThemeMenuOpen(false);
  };

  const handleLoginSuccess = () => {
    setIsLoggedIn(true);
    setPage("map");
  };

  const handleLogoutSuccess = () => {
    clearAuthToken();
    setIsLoggedIn(false);
    setPage("map");
  };

  const getButtonClass = (targetPage) =>
    `side-nav-btn ${page === targetPage ? "active" : ""}`;

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
        {/* ✅ 로고 */}
        <div className="side-logo" onClick={() => setPage("map")}>
          <img src={DISH_LOGO} alt="DISHINSIDE 로고" className="side-logo-img" />
        </div>

        {/* ✅ MAP 페이지 카테고리 드롭다운 */}
        {page === "map" && (
          <div className="side-category-card">
            <div className="side-category-title">카테고리</div>
            <select
              className="side-category-select"
              value={categoryFilterId}
              onChange={(e) => setCategoryFilterId(e.target.value)}
            >
              <option value="">전체</option>
              {CATEGORIES.map((c) => (
                <option key={c.id} value={String(c.id)}>
                  {c.label}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* ✅ FAVORITE 페이지 카테고리 드롭다운 (헤더에서 내려옴) */}
        {page === "favorite" && (
          <div className="side-category-card">
            <div className="side-category-title">카테고리</div>
            <select
              className="side-category-select"
              value={favoriteCategoryFilter}
              onChange={(e) => setFavoriteCategoryFilter(e.target.value)}
            >
              {FAVORITE_CATEGORY_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        )}

        <button
          className={getButtonClass("community")}
          onClick={() => setPage("community")}
        >
          커뮤니티
        </button>

        <button className={getButtonClass("profile")} onClick={() => setPage("profile")}>
          회원정보
        </button>

        <button
          className={getButtonClass("favorite")}
          onClick={() => setPage("favorite")}
        >
          즐겨찾기
        </button>

        {/* ✅ 좌하단 설정 + 로그아웃 */}
        <div className="side-settings-wrap">
          <button
            type="button"
            className="side-settings-btn"
            onClick={() => setIsThemeMenuOpen((prev) => !prev)}
            title="테마 설정"
          >
            <img src={SETTING_ICON} alt="테마 설정" className="side-settings-icon" />
          </button>

          <button type="button" className="side-logout-mini" onClick={() => setPage("logout")}>
            로그아웃
          </button>

          {isThemeMenuOpen && (
            <div className="side-settings-menu">
              <button
                type="button"
                className={"side-settings-item" + (theme === "light" ? " active" : "")}
                onClick={setLight}
              >
                라이트 모드
              </button>
              <button
                type="button"
                className={"side-settings-item" + (theme === "dark" ? " active" : "")}
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
          <KakaoMap categoryFilterId={categoryFilterId} />
        ) : page === "community" ? (
          <CommunityPage />
        ) : page === "profile" ? (
          <UserProfilePage onWithdraw={() => setPage("withdraw")} />
        ) : page === "favorite" ? (
          <FavoritePage categoryFilter={favoriteCategoryFilter} />
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
