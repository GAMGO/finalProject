// src/components/Sidebar.jsx
import { useState } from "react";
import logo from "../assets/logo.png";
import setIcon from "../assets/set.png";
import newIcon from "../assets/new.png";
import trashIcon from "../assets/trash.png";

export default function Sidebar({
  open, // true: 펼침 / false: 접힘
  onToggle,
  theme,
  onChangeTheme,
  sessions,
  currentSessionId,
  onCreateSession,
  onSelectSession,
  onDeleteSession, // 🔴 삭제 콜백 (App에서 내려줌)
}) {
  const [showTheme, setShowTheme] = useState(false);
  const [openMenuId, setOpenMenuId] = useState(null); // 어떤 세션의 … 메뉴가 열려 있는지

  const handleThemeClick = () => setShowTheme((v) => !v);
  const handleThemeSelect = (value) => {
    onChangeTheme(value);
    setShowTheme(false);
  };

  const toggleSessionMenu = (e, sessionId) => {
    e.stopPropagation();
    setOpenMenuId((prev) => (prev === sessionId ? null : sessionId));
  };

  const handleDeleteClick = (e, sessionId) => {
    e.stopPropagation();
    setOpenMenuId(null);
    if (window.confirm("이 대화를 삭제할까요?")) {
      onDeleteSession?.(sessionId);
    }
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

      {/* 접힌 상태에서만: 새 대화 아이콘 */}
      {!open && (
        <button
          type="button"
          className="collapsed-new-btn"
          onClick={onCreateSession}
        >
          <img src={newIcon} alt="새 대화" className="collapsed-new-icon" />
        </button>
      )}

      {/* 펼쳐진 상태에서만: 새 대화, 세션 리스트, 테마 */}
      {open && (
        <>
          {/* + 새 대화 버튼 */}
          <div className="sidebar-new">
            <button
              type="button"
              className="btn-new-chat"
              onClick={onCreateSession}
            >
              + 새 대화
            </button>
          </div>

          {/* 세션 리스트 */}
          <div className="sidebar-sessions">
            {sessions.length === 0 ? (
              <div className="session-empty" />
            ) : (
              sessions.map((s) => {
                const isActive = s.id === currentSessionId;
                const isMenuOpen = openMenuId === s.id;

                return (
                  <div
                    key={s.id}
                    className={
                      "session-item" + (isActive ? " active" : "")
                    }
                  >
                    {/* 메인 영역 (제목/시간) */}
                    <button
                      type="button"
                      className="session-main"
                      onClick={() => onSelectSession(s.id)}
                    >
                      <div className="session-title">{s.title}</div>
                      <div className="session-time">{s.timeLabel}</div>
                    </button>

                    {/* … 버튼 */}
                    <button
                      type="button"
                      className="session-menu-btn"
                      onClick={(e) => toggleSessionMenu(e, s.id)}
                    >
                      ⋯
                    </button>

                    {/* 삭제 말풍선 */}
                    {isMenuOpen && (
                      <div className="session-menu">
                        <button
                          type="button"
                          className="session-menu-delete"
                          onClick={(e) => handleDeleteClick(e, s.id)}
                        >
                          <img
                            src={trashIcon}
                            alt="삭제"
                            className="session-menu-icon"
                          />
                          삭제
                        </button>
                      </div>
                    )}
                  </div>
                );
              })
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
