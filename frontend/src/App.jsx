import { useEffect, useState } from "react";
import { api } from "./api";
import ChatScreen from "./ChatScreen";

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);

  // 세션 목록 로드
  useEffect(() => {
    api.get("/chat/sessions").then((res) => {
      setSessions(res.data);
      if (res.data.length > 0 && !currentSessionId) {
        setCurrentSessionId(res.data[0].id);
      }
    });
  }, []);

  const createSession = async () => {
    const title = prompt("새 대화 제목 (예: 출장비 양식)");
    if (!title) return;
    const res = await api.post("/chat/sessions", { title });
    setSessions((prev) => [res.data, ...prev]);
    setCurrentSessionId(res.data.id);
  };

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* 왼쪽: 세션 리스트 */}
      <div
        style={{
          width: 260,
          borderRight: "1px solid #ddd",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div style={{ padding: 12, borderBottom: "1px solid #eee" }}>
          <button onClick={createSession} style={{ width: "100%" }}>
            + 새 대화
          </button>
        </div>
        <div style={{ flex: 1, overflowY: "auto" }}>
          {sessions.map((s) => (
            <div
              key={s.id}
              onClick={() => setCurrentSessionId(s.id)}
              style={{
                padding: "8px 12px",
                cursor: "pointer",
                backgroundColor:
                  s.id === currentSessionId ? "#f4f4f4" : "transparent",
              }}
            >
              <div style={{ fontSize: 14, fontWeight: 600 }}>{s.title}</div>
              <div style={{ fontSize: 11, color: "#666" }}>{s.updatedAt}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 오른쪽: 선택된 세션 채팅 */}
      <div style={{ flex: 1 }}>
        {currentSessionId ? (
          <ChatScreen sessionId={currentSessionId} />
        ) : (
          <div style={{ padding: 24 }}>왼쪽에서 새 대화를 만들어 주세요.</div>
        )}
      </div>
    </div>
  );
}
