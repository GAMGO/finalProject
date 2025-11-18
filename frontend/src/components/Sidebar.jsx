// src/components/Sidebar.jsx
import { useState } from "react";
import logo from "../assets/logo.png";
import setIcon from "../assets/set.png";
import newIcon from "../assets/new.png";

export default function Sidebar({
  open,                 // true: 펼침 / false: 접힘
  onToggle,             // 로고 클릭 시 접기/펼치기
  theme,
  onChangeTheme,
  sessions,
  currentSessionId,
  onCreateSession,      // ✅ App에서 온 모달 오픈 함수
  onSelectSession,
}) {
  const [showTheme, setShowTheme] = useState(false);

  const handleThemeClick = () => setShowTheme((v) => !v);
  const handleThemeSelect = (value) => {
    onChangeTheme(value);
    setShowTheme(false);
  };

  return (
    <aside className={`sidebar ${open ? "" : "collapsed"}`}>
      {/* 로고 행 - 클릭하면 접기/펼치기 */}
      <div className="sidebar-logo-row" onClick={onToggle}>
        <img src={logo} className="sidebar-logo" alt="Excel AI 비서" />
        {open && (
          <div className="sidebar-logo-text">
            <div className="sidebar-logo-title">Excel AI 비서</div>
            <div className="sidebar-logo-sub">클릭하면 접기/펼치기</div>
          </div>
        )}
      </div>

      {/* ✅ 접힌 상태에서만 보이는 새 대화 아이콘 */}
      {!open && (
        <button
          type="button"
          className="collapsed-new-btn"
          onClick={onCreateSession}   // ← 여기서 모달 열기
        >
          <img src={newIcon} alt="새 대화" className="collapsed-new-icon" />
        </button>
      )}

      {/* 펼쳐진 상태에서만: 새 대화 버튼 + 세션 리스트 + 테마 설정 */}
      {open && (
        <>
          {/* + 새 대화 버튼 */}
          <div className="sidebar-new">
            <button
              type="button"
              className="btn-new-chat"
              onClick={onCreateSession}   // ← 여기도 같은 모달
            >
              + 새 대화
            </button>
          </div>

          {/* 세션 리스트 */}
          <div className="sidebar-sessions">
            {sessions.length === 0 ? (
              <div className="session-empty">
                이전에 했던 대화는
                <br />
                위 로고를 클릭해서
                <br />
                히스토리에서 불러올 수 있어요.
              </div>
            ) : (
              sessions.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  className={
                    "session-item" +
                    (s.id === currentSessionId ? " active" : "")
                  }
                  onClick={() => onSelectSession(s.id)}
                >
                  <div className="session-title">{s.title}</div>
                  <div className="session-time">{s.createdAtLabel}</div>
                </button>
              ))
            )}
          </div>

          {/* 하단 테마 설정 */}
          <div className="sidebar-footer">
            <button
              type="button"
              className="settings-btn"
              onClick={handleThemeClick}
            >
              <img src={setIcon} alt="테마 설정" className="settings-icon" />
            </button>

            {showTheme && (
              <div className="theme-dropdown">
                <button
                  type="button"
                  className={
                    "theme-item" + (theme === "light" ? " active" : "")
                  }
                  onClick={() => handleThemeSelect("light")}
                >
                  라이트
                </button>
                <button
                  type="button"
                  className={
                    "theme-item" + (theme === "dark" ? " active" : "")
                  }
                  onClick={() => handleThemeSelect("dark")}
                >
                  다크
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
