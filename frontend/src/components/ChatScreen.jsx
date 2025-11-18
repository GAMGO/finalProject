import { useEffect, useState } from "react";
import { api } from "../api";
import "../styles/Chat.css";
import graySend from "../assets/gray_send.png";
import greenSend from "../assets/green_send.png";

export default function ChatScreen({ sessionId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // 세션 변경 시 메시지 로드
  useEffect(() => {
    if (!sessionId) return;
    api
      .get(`/chat/sessions/${sessionId}/messages`)
      .then((res) => setMessages(res.data || []))
      .catch((err) => console.error(err));
  }, [sessionId]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || !sessionId || loading) return;

    setInput("");
    setLoading(true);

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
          content:
            "메시지 전송 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요.",
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

  const isEmpty = messages.length === 0;
  const canSend = !!input.trim() && !loading;

  return (
    <div className="chat-root">
      {/* 채팅 본문 */}
      <div className="chat-body">
        <div className="chat-body-inner">
          {/* 첫 대화 없을 때 웰컴 블록 */}
          {isEmpty && (
            <div className="welcome-block">
              <div className="welcome-title">엑셀 비서에요</div>
              <div className="welcome-desc">
                만들고 싶은{" "}
                <span className="welcome-highlight">엑셀 양식</span>을
                자연스럽게 말해 주세요.
                <br />
                저는 형식을 먼저 제안하고,{" "}
                <b>“이 형식으로 만들어드릴까요?”</b>처럼 꼭 확인을 받고
                진행해요.
              </div>
              <ul className="welcome-list">
                <li>출장비 정리용 엑셀 만들고 싶어</li>
                <li>주간 근무표 양식 깔끔하게 새로 만들고 싶어</li>
                <li>매출·원가·이익이 한 눈에 보이게 보고서 틀 만들어줘</li>
              </ul>
            </div>
          )}

          {/* 메시지 리스트 */}
          {messages.map((m) => (
            <div
              key={m.id}
              className={`chat-message-row ${
                m.sender === "USER" ? "me" : "ai"
              }`}
            >
              <div className="chat-bubble">
                <div className="chat-bubble-label">
                  {m.sender === "USER" ? "나" : "AI 비서"}
                </div>
                <div className="chat-bubble-text">{m.content}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 입력 영역 */}
      <div className="chat-footer">
        <div className="chat-input-row">
          <textarea
            rows={2}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="엑셀로 만들 업무를 설명해 주세요. 예: 출장비 정리용 엑셀 만들어줘"
          />
          <button
            type="button"
            onClick={sendMessage}
            disabled={!canSend}
            className="send-btn"
            aria-label="메시지 전송"
          >
            <img
              src={canSend ? greenSend : graySend}
              alt=""
              className="send-icon"
            />
          </button>
        </div>
        <div className="chat-hint">
        </div>
      </div>
    </div>
  );
}
