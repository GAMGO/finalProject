// src/App.js
import React, { useState } from "react";
import KakaoMap from "./components/KakaoMap";
import { LoginForm } from "./components/LoginForm";
import { SettingsPage } from "./components/SettingsPage";
import { ResetPasswordByQuestion } from "./components/ResetPasswordByQuestion";
import "./App.css";

export default function App() {
  // 로그인 여부
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  // 사이드바에서 어떤 화면을 보여줄지
  // 'login' | 'settings' | 'reset'
  const [view, setView] = useState("login");

  const handleLoggedIn = () => {
    setIsLoggedIn(true);
    setView("settings");
  };

  const handleLoggedOut = () => {
    setIsLoggedIn(false);
    setView("login");
  };

  return (
    <div className="app">
      <aside className="side-bar">
        <div className="side-logo">LOGO</div>

        <div className="side-nav">
          {/* 로그인 안 되어 있으면 로그인/비번찾기 메뉴만 */}
          {!isLoggedIn && (
            <>
              {view !== "login" && (
                <button onClick={() => setView("login")}>로그인</button>
              )}
              {view !== "reset" && (
                <button onClick={() => setView("reset")}>비밀번호 찾기</button>
              )}
            </>
          )}

          {/* 로그인 되어 있으면 설정 메뉴만 */}
          {isLoggedIn && view !== "settings" && (
            <button onClick={() => setView("settings")}>설정</button>
          )}
        </div>

        <div className="side-content">
          {view === "login" && (
            <LoginForm onLoggedIn={handleLoggedIn} />
          )}

          {view === "reset" && (
            <ResetPasswordByQuestion />
          )}

          {view === "settings" && isLoggedIn && (
            <SettingsPage onLogout={handleLoggedOut} />
          )}

          {/* 혹시 로그인 안 된 상태에서 settings 눌렀을 때 보호 */}
          {view === "settings" && !isLoggedIn && (
            <div>로그인이 필요합니다.</div>
          )}
        </div>
      </aside>

      <main className="main-content">
        <KakaoMap />
      </main>
    </div>
  );
}

/*
 * [파일 설명]
 * - 좌측 사이드바 + 우측 카카오맵 레이아웃의 메인 App 컴포넌트.
 * - 사이드바:
 *   - 로그인 전: LoginForm, 비밀번호 찾기(보안질문) 화면 전환.
 *   - 로그인 후: SettingsPage(회원정보, 로그인 기록, 비번 변경, 보안질문 설정) 표시.
 * - 메인 영역(main-content)은 항상 KakaoMap만 렌더링.
 */
