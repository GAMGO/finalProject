// src/App.jsx
import { useEffect, useState } from "react";
import "./styles/App.css";
import "./styles/Chat.css";
import "./styles/Sidebar.css";
import Sidebar from "./components/Sidebar";
import ChatScreen from "./components/ChatScreen";
import NewSessionModal from "./components/NewSessionModal";
import { api } from "./api";
import Signup from "./components/SignUp";
import { useLocation } from "react-router-dom";
export default function App() {
  const [theme, setTheme] = useState("light");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [currentTitle, setCurrentTitle] = useState("");

  const [newModalOpen, setNewModalOpen] = useState(false);
  const location = useLocation();

  // 세션 목록 처음 로딩
  useEffect(() => {
    api
      .get("/chat/sessions")
      .then((res) => {
        const raw = res.data || [];

        // 👉 어떤 이름으로 오든 timeLabel 하나로 통일
        const list = raw.map((s) => ({
          ...s,
          timeLabel: s.updatedAt || s.createdAtLabel || s.createdAt || "", // 혹시 없으면 빈 문자열
        }));

        setSessions(list);

        if (list.length && !currentSessionId) {
          setCurrentSessionId(list[0].id);
          setCurrentTitle(list[0].title);
        }
      })
      .catch((err) => console.error(err));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 현재 세션 바뀔 때마다 제목 업데이트
  useEffect(() => {
    const s = sessions.find((x) => x.id === currentSessionId);
    setCurrentTitle(s ? s.title : "");
  }, [sessions, currentSessionId]);

  // "+ 새 대화" 버튼 / 접힌 상태 newIcon 클릭 → 모달 열기
  const openNewSessionModal = () => {
    setNewModalOpen(true);
  };

  // 모달에서 "대화 만들기" 눌렀을 때
  const handleCreateSessionConfirm = async (title) => {
    const trimmed = title.trim();
    if (!trimmed) return;

    try {
      const res = await api.post("/chat/sessions", { title: trimmed });
      const dto = res.data;

      const newSession = {
        ...dto,
        timeLabel: dto.updatedAt || dto.createdAtLabel || dto.createdAt || "",
      };

      // 새 세션을 목록 맨 위에 추가
      setSessions((prev) => [newSession, ...prev]);
      setCurrentSessionId(newSession.id);
      setNewModalOpen(false);
    } catch (e) {
      console.error(e);
      alert("새 대화 생성 중 오류가 발생했어요.");
    }
  };

  // 세션 선택
  const handleSelectSession = (id) => {
    setCurrentSessionId(id);
  };

  // 세션 삭제
  const handleDeleteSession = async (id) => {
    const target = sessions.find((s) => s.id === id);
    if (!target) return;

    // ✅ 여기서만 confirm 처리 (Sidebar 쪽 confirm은 지우는 걸 추천)
    const ok = window.confirm(`"${target.title}" 대화를 삭제할까요?`);
    if (!ok) return;

    try {
      await api.delete(`/chat/sessions/${id}`);

      // 상태에서 제거 + 현재 세션이면 옮겨가기
      setSessions((prev) => {
        const next = prev.filter((s) => s.id !== id);

        if (id === currentSessionId) {
          if (next.length > 0) {
            setCurrentSessionId(next[0].id);
          } else {
            setCurrentSessionId(null);
            setCurrentTitle("");
          }
        }

        return next;
      });
    } catch (e) {
      console.error(e);
      alert("대화 삭제 중 오류가 발생했어요.");
    }
  };

  return location.pathname === "/signup" ? (
    <Signup />
  ) : (
    <div className={`app-root ${theme}`}>
      {/* 사이드바 */}
      <Sidebar
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((v) => !v)}
        theme={theme}
        onChangeTheme={setTheme}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onCreateSession={openNewSessionModal}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
      />

      {/* 메인 영역 */}
      <div className="main-area">
        {/* 상단 헤더 (제목 표시) */}
        <header className="chat-header">
          <div className="chat-header-inner">
            <div className="chat-header-title">{currentTitle || "새 대화"}</div>
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
            <span className="main-empty-highlight">왼쪽에서 “+ 새 대화”</span>를
            눌러 새 엑셀 비서 대화를 시작해 보세요.
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
