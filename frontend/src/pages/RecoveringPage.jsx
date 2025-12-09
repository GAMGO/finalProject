import React, { useState } from "react";
import axios from "axios";

const baseURL = import.meta.env.VITE_BASE_URL;

const RecoveringPage = ({ onAuthSuccess }) => {
  const [token, setToken] = useState("");
  const [message, setMessage] = useState({ text: "", type: "" });

  const handleRestore = async () => {
    try {
      await axios.post(`${baseURL}/api/auth/restore`, { recoveryToken: token });
      setMessage({ text: "복구 성공! 로그인 페이지로 이동합니다.", type: "success" });
      setTimeout(() => onAuthSuccess(), 2000);
    } catch (error) {
      setMessage({ text: "유효하지 않은 코드입니다.", type: "error" });
    }
  };

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