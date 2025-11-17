import { useEffect, useState } from "react";
import { api } from "./api";

export default function ChatScreen({ sessionId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // 세션 변경 시 메시지 로드
  useEffect(() => {
    if (!sessionId) return;
    api
      .get(`/chat/sessions/${sessionId}/messages`)
      .then((res) => setMessages(res.data))
      .catch((err) => console.error(err));
  }, [sessionId]);

  const sendMessage = async () => {
    if (!input.trim() || !sessionId) return;

    const text = input;
    setInput("");
    setLoading(true);

    // 화면에서 바로 보이게 임시 메시지 넣기
    const tempId = Date.now();
    const tempUser = {
      id: tempId,
      sender: "USER",
      content: text,
    };
    setMessages((prev) => [...prev, tempUser]);

    try {
      const res = await api.post(`/chat/sessions/${sessionId}/messages`, {
        content: text,
      });
      const { userMessage, assistantMessage } = res.data;

      // 임시 메시지 제거하고 실제 저장된 메시지로 교체
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== tempId),
        userMessage,
        assistantMessage,
      ]);
    } catch (e) {
      console.error(e);
      setMessages((prev) => [
        ...prev,
        {
          id: tempId + 1,
          sender: "AI",
          content: "메시지 전송 중 오류가 발생했어요. 다시 시도해 주세요.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* 채팅 로그 */}
      <div style={{ flex: 1, padding: 16, overflowY: "auto" }}>
        {messages.map((m) => (
          <div
            key={m.id}
            style={{
              marginBottom: 8,
              textAlign: m.sender === "USER" ? "right" : "left",
            }}
          >
            <div
              style={{
                display: "inline-block",
                padding: "8px 12px",
                borderRadius: 8,
                backgroundColor: m.sender === "USER" ? "#d1e7ff" : "#f1f1f1",
              }}
            >
              <div style={{ fontSize: 12, opacity: 0.7 }}>
                {m.sender === "USER" ? "나" : "AI 비서"}
              </div>
              <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>
            </div>
          </div>
        ))}
      </div>

      {/* 입력창 */}
      <div style={{ padding: 16, borderTop: "1px solid #eee" }}>
        <textarea
          rows={3}
          style={{ width: "100%", resize: "none", marginBottom: 8 }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="업무를 설명해 주세요. 예: 출장비 정리용 엑셀 만들어줘"
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? "생각 중..." : "전송"}
        </button>
      </div>
    </div>
  );
}
