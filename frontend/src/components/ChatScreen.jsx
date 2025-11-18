// src/components/ChatScreen.jsx
import { useEffect, useState, useRef } from "react";
import { api } from "../api";
import "../styles/Chat.css";
import graySend from "../assets/gray_send.png";
import greenSend from "../assets/green_send.png";
import uploadIcon from "../assets/upload.png";
import fileIcon from "../assets/file.png";
import imageIcon from "../assets/image.png";
import downloadIcon from "../assets/download.png"; // ⬅ 다운로드 아이콘

const FILE_BASE = "http://localhost:8080"; // 업로드 파일 불러올 때 사용

export default function ChatScreen({ sessionId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const [showUploadMenu, setShowUploadMenu] = useState(false);
  const [uploading, setUploading] = useState(false);

  // ⭐ 이미지 미리보기 상태
  const [previewImage, setPreviewImage] = useState(null); // { url, name }

  const fileInputRef = useRef(null);
  const imageInputRef = useRef(null);

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
    if (!text || !sessionId || loading || uploading) return;

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

  /* ======================
     업로드 관련 핸들러
     ====================== */

  const toggleUploadMenu = () => {
    setShowUploadMenu((v) => !v);
  };

  const handleSelectFile = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const handleSelectImage = () => {
    if (imageInputRef.current) imageInputRef.current.click();
  };

  const handleUpload = async (e, type) => {
    const file = e.target.files?.[0];
    if (!file || !sessionId) return;

    // 같은 파일 다시 선택할 수 있게 value 초기화
    e.target.value = "";

    setUploading(true);
    setShowUploadMenu(false);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("type", type);

    try {
      const res = await api.post(
        `/chat/sessions/${sessionId}/upload`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      const msg = res.data;
      setMessages((prev) => [...prev, msg]);
    } catch (err) {
      console.error(err);
      alert("파일 업로드 중 오류가 발생했어요.");
    } finally {
      setUploading(false);
    }
  };

  const isEmpty = messages.length === 0;
  const canSend = !!input.trim() && !loading && !uploading;

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
          {messages.map((m) => {
            const hasFile = !!m.fileUrl;
            const fileUrl = hasFile ? `${FILE_BASE}${m.fileUrl}` : null;

            return (
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
                  <div className="chat-bubble-text">
                    {hasFile ? (
                      m.fileType === "IMAGE" ? (
                        // 이미지: 썸네일 클릭 → 미리보기 모달
                        <button
                          type="button"
                          className="image-thumb-btn"
                          onClick={() =>
                            setPreviewImage({
                              url: fileUrl,
                              name: m.fileName || "이미지",
                            })
                          }
                        >
                          <img
                            src={fileUrl}
                            alt={m.fileName || "이미지"}
                            className="chat-image-preview"
                          />
                        </button>
                      ) : (
                        // 일반 파일: 카드 클릭 시 바로 다운로드
                        <a
                          href={fileUrl}
                          className="chat-file-link"
                          download={m.fileName || true}
                        >
                          <div className="file-card">
                            <div className="file-card-icon-wrapper">
                              <img
                                src={fileIcon}
                                alt=""
                                className="file-card-icon"
                              />
                            </div>
                            <div className="file-card-text">
                              <div className="file-card-name">
                                {m.fileName || "파일"}
                              </div>
                              <div className="file-card-sub">파일</div>
                            </div>
                          </div>
                        </a>
                      )
                    ) : (
                      m.content
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 입력 영역 */}
      <div className="chat-footer">
        <div className="chat-input-row">
          {/* ⬅ 업로드 버튼 (왼쪽) */}
          <div className="upload-wrapper">
            <button
              type="button"
              className="upload-btn"
              onClick={toggleUploadMenu}
              aria-label="파일/이미지 업로드"
            >
              <img src={uploadIcon} alt="" className="upload-icon" />
            </button>

            {showUploadMenu && (
              <div className="upload-menu">
                <button
                  type="button"
                  className="upload-menu-item"
                  onClick={handleSelectFile}
                >
                  <img src={fileIcon} alt="" className="upload-menu-icon" />
                  <span>파일 업로드</span>
                </button>
                <button
                  type="button"
                  className="upload-menu-item"
                  onClick={handleSelectImage}
                >
                  <img src={imageIcon} alt="" className="upload-menu-icon" />
                  <span>이미지 업로드</span>
                </button>
              </div>
            )}
          </div>

          {/* 가운데 텍스트 입력 */}
          <textarea
            rows={2}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="엑셀로 만들 업무를 설명해 주세요. 예: 출장비 정리용 엑셀 만들어줘"
          />

          {/* 오른쪽 전송 버튼 */}
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

        <div className="chat-hint"></div>

        <input
          type="file"
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={(e) => handleUpload(e, "FILE")}
        />
        <input
          type="file"
          ref={imageInputRef}
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => handleUpload(e, "IMAGE")}
        />
      </div>

      {/* 이미지 미리보기 모달 */}
      {previewImage && (
        <div
          className="image-preview-backdrop"
          onClick={() => setPreviewImage(null)}
        >
          <div
            className="image-preview-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={previewImage.url}
              alt={previewImage.name}
              className="image-preview-large"
            />
            <div className="image-preview-footer">
              <div className="image-preview-name">{previewImage.name}</div>
              <a
                href={previewImage.url}
                className="image-preview-download-btn"
                download={previewImage.name || "image"}
              >
                <img
                  src={downloadIcon}
                  alt="다운로드"
                  className="image-preview-download-icon"
                />
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
