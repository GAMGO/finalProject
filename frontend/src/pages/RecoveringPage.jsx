import React, { useState, useEffect } from "react";
import axios from "axios";

const baseURL = import.meta.env.VITE_LOCAL_BASE_URL;

const RecoveringPage = ({ onAuthSuccess }) => {
  // URL 쿼리 파라미터에서 토큰을 추출하여 초기값으로 설정
  const initialToken = new URLSearchParams(window.location.search).get("token") || "";
  
  const [token, setToken] = useState(initialToken);
  const [message, setMessage] = useState({ text: "", type: "" });

  const handleRestore = async () => {
    // 토큰이 없거나 6자리가 아니면 복구 시도를 막음 (선택적)
    if (!token || token.length !== 6) {
        setMessage({ text: "유효하지 않은 코드 형식입니다.", type: "error" });
        return;
    }

    try {
      await axios.post(`${baseURL}/api/auth/restore`, { recoveryToken: token });
      setMessage({ text: "복구 성공! 로그인 페이지로 이동합니다.", type: "success" });
      setTimeout(() => onAuthSuccess(), 2000);
    } catch (error) {
      // 서버에서 400 Bad Request 등을 반환할 때
      setMessage({ text: "유효하지 않은 코드입니다.", type: "error" });
    }
  };

  // URL 파라미터에 토큰이 있을 경우, 자동으로 복구 시도 (선택적)
   useEffect(() => {
       if (initialToken && initialToken.length === 6) {
           handleRestore();
       }
  }, []);

  const boxStyle = { backgroundColor: "#F5D7B7", padding: "60px 40px", borderRadius: "40px", width: "45vh", textAlign: "center" };
  const inputStyle = { width: "100%", padding: "12px", borderRadius: "20px", border: "none", boxShadow: `4px 4px 0px #78266A`, textAlign: "center", fontSize: "20px", letterSpacing: "5px" };
  const buttonStyle = { backgroundColor: "#FFFFFF", color: "#78266A", padding: "10px 30px", borderRadius: "20px", border: "2px solid #78266A", marginTop: "20px", cursor: "pointer", boxShadow: "4px 4px 0px #78266A" };

  return (
    <div style={boxStyle}>
      <h2 style={{ color: "#78266A" }}>계정 복구</h2>
      <p style={{ color: "#78266A", fontSize: "14px", marginBottom: "20px" }}>메일로 발송된 6자리 복구 코드를 입력해주세요.</p>
      <input
        style={inputStyle}
        maxLength={6}
        value={token}
        onChange={(e) => setToken(e.target.value)}
      />
      <button style={buttonStyle} onClick={handleRestore}>복구하기</button>
      {message.text && <div style={{ marginTop: "15px", padding: "10px", backgroundColor: message.type === "error" ? "#D9534F" : "#5CB85C", color: "white", borderRadius: "10px" }}>{message.text}</div>}
    </div>
  );
};

export default RecoveringPage;