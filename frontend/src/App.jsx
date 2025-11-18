// src/App.jsx
import { useEffect, useState } from "react";
import "./styles/App.css";
import "./styles/Chat.css";
import "./styles/Sidebar.css";
import Sidebar from "./components/Sidebar";
import ChatScreen from "./components/ChatScreen";
import NewSessionModal from "./components/NewSessionModal";
import { api } from "./api";

export default function App() {
  const [theme, setTheme] = useState("light");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [currentTitle, setCurrentTitle] = useState("");

  const [newModalOpen, setNewModalOpen] = useState(false);

  // 세션 목록 처음 로딩
  useEffect(() => {
    api
      .get("/chat/sessions")
      .then((res) => {
        const list = res.data || [];
        setSessions(list);
        if (list.length && !currentSessionId) {
          setCurrentSessionId(list[0].id);
          setCurrentTitle(list[0].title);
        }
      })
      .catch((err) => console.error(err));
  }, []);

  // 현재 세션 바뀔 때마다 제목 업데이트
  useEffect(() => {
    const s = sessions.find((x) => x.id === currentSessionId);
    setCurrentTitle(s ? s.title : "");
  }, [sessions, currentSessionId]);

  // "+ 새 대화" 버튼 클릭 → 모달 열기
  const openNewSessionModal = () => {
    setNewModalOpen(true);
  };

  // 모달에서 "대화 만들기" 눌렀을 때
  const handleCreateSessionConfirm = async (title) => {
    try {
      const res = await api.post("/chat/sessions", { title: title.trim() });
      const newSession = res.data;
      // 새 세션을 목록 맨 위에 추가
      setSessions((prev) => [newSession, ...prev]);
      setCurrentSessionId(newSession.id);
      setNewModalOpen(false);
    } catch (e) {
      console.error(e);
      alert("새 대화 생성 중 오류가 발생했어요.");
    }
  };

  const handleSelectSession = (id) => {
    setCurrentSessionId(id);
  };

  return (
    <div className={`app-root ${theme}`}>
      {/* 사이드바 */}
      <Sidebar
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((v) => !v)}
        theme={theme}
        onChangeTheme={setTheme}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onCreateSession={openNewSessionModal} // ✅ 여기서 모달 오픈
        onSelectSession={handleSelectSession}
      />

      {/* 메인 영역 */}
      <div className="main-area">
        {/* 상단 헤더 (제목 표시) */}
        <header className="chat-header">
          <div className="chat-header-inner">
            <div className="chat-header-title">
              {currentTitle || "새 대화"}
            </div>
            <div className="chat-header-sub">
              엑셀 양식 자동화를 위한 AI 비서
            </div>
          </div>
        </header>

        {/* 본문 */}
        {currentSessionId ? (
          <ChatScreen sessionId={currentSessionId} />
        ) : (
          <div className="main-empty">
            <span className="main-empty-highlight">왼쪽에서 “+ 새 대화”</span>
            를 눌러 새 엑셀 비서 대화를 시작해 보세요.
          </div>
        )}
      </div>

      {/* 새 대화 제목 모달 */}
      <NewSessionModal
        open={newModalOpen}
        onClose={() => setNewModalOpen(false)}
        onConfirm={handleCreateSessionConfirm}
      />
    </div>
  );
}
