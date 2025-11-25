import React, { useState } from "react";
import KakaoMap from "./components/KakaoMap";
import CommunityPage from "./pages/CommunityPage";
import UserProfilePage from "./pages/UserProfilePage";   // ✅ 추가
import FavoritePage from "./pages/FavoritePage";         // ✅ 추가
import "./App.css";
import "./SidebarPatch.css";

export default function App() {
<<<<<<< HEAD
  const [page, setPage] = useState("map"); // map / community
=======
  const [page, setPage] = useState("map"); // map / community / profile / favorite

>>>>>>> d74e9d5a1db6651300ad0dde6709c0491e381714
  return (
    <div className="app">
      <aside className="side-bar">
        <div className="side-logo" onClick={() => setPage("map")}>
        </div>

        <button
          className="side-nav-btn"
          onClick={() => setPage("community")}
        >
          커뮤니티
        </button>

        {/* ✅ 여기부터 “추가만” */}
        <button
          className="side-nav-btn"
          onClick={() => setPage("profile")}
        >
          회원정보
        </button>

        <button
          className="side-nav-btn"
          onClick={() => setPage("favorite")}
        >
          즐겨찾기
        </button>
        {/* ✅ 여기까지 */}
      </aside>

      <main className="main-content">
        {page === "map" ? (
          <KakaoMap />
        ) : page === "community" ? (
          <CommunityPage />
        ) : page === "profile" ? (
          <UserProfilePage />
        ) : (
          <FavoritePage />
        )}
      </main>
    </div>
  );
}