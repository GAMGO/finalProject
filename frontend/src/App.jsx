import React, { useState } from "react";
import KakaoMap from "./components/KakaoMap";
import CommunityPage from "./pages/CommunityPage";
import "./App.css";

export default function App() {
  const [page, setPage] = useState("map"); // map / community
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
      </aside>
      <main className="main-content">
        {page === "map" ? <KakaoMap /> : <CommunityPage />}
      </main>
    </div>
  );
}