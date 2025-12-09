import React, { useState, useEffect } from "react";
import axios from "axios";
import "../theme/theme.css";

const baseURL = import.meta.env.VITE_LOCAL_BASE_URL;

const WithdrawalPage = ({ userId, onLogout }) => {
  const [checks, setChecks] = useState({ info: false, email: false, law: false });
  const [confirmText, setConfirmText] = useState("");
  const [message, setMessage] = useState({ text: "", type: "" });

  const isAllChecked = checks.info && checks.email && checks.law;
  const targetText = `${userId} 회원 탈퇴를 진행하는 것에 동의합니다.`;

  const handleCheck = (e) => {
    const { name, checked } = e.target;
    setChecks((prev) => ({ ...prev, [name]: checked }));
  };

  const handleWithdraw = async () => {
    if (confirmText !== targetText) return;

    try {
      const token = localStorage.getItem("jwtToken");
      await axios.delete(`${baseURL}/api/auth/withdrawal`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMessage({ text: "탈퇴 처리가 완료되었습니다. 이메일을 확인해주세요.", type: "success" });
      setTimeout(() => onLogout(), 2000); // 탈퇴 후 로그아웃 처리
    } catch (error) {
      setMessage({ text: "탈퇴 처리 중 오류가 발생했습니다.", type: "error" });
    }
  };

  // 스타일 정의 (LoginPage 스타일 계승)
  const darkPurple = "#78266A";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";

  const boxStyle = {
    backgroundColor: lightPeach,
    padding: "40px",
    borderRadius: "40px",
    width: "45vh",
    textAlign: "center",
    fontFamily: "PartialSans, sans-serif",
  };

  const checkGroupStyle = {
    textAlign: "left",
    marginBottom: "15px",
    color: darkPurple,
    fontSize: "14px",
    lineHeight: "1.6",
  };

  const inputStyle = {
    width: "100%",
    padding: "12px",
    borderRadius: "20px",
    border: "none",
    marginTop: "10px",
    boxShadow: `4px 4px 0px ${darkPurple}`,
  };

  const buttonStyle = {
    backgroundColor: white,
    color: darkPurple,
    padding: "10px 30px",
    borderRadius: "20px",
    border: `2px solid ${darkPurple}`,
    cursor: confirmText === targetText ? "pointer" : "not-allowed",
    marginTop: "20px",
    opacity: confirmText === targetText ? 1 : 0.5,
    boxShadow: `4px 4px 0px ${darkPurple}`,
  };

  return (
    <div style={boxStyle}>
      <h2 style={{ color: darkPurple, marginBottom: "20px" }}>회원 탈퇴</h2>
      
      <div style={checkGroupStyle}>
        <input type="checkbox" name="info" onChange={handleCheck} /> 1. 개인정보 파기 동의: 즉시 파기됨을 확인합니다.
      </div>
      <div style={checkGroupStyle}>
        <input type="checkbox" name="email" onChange={handleCheck} /> 2. 탈퇴 시 이메일 알림 수신에 동의합니다.
      </div>
      <div style={checkGroupStyle}>
        <input type="checkbox" name="law" onChange={handleCheck} /> 3. 법령에 의거한 일정 기간 보관 후 파기에 동의합니다.
      </div>

      {/* 부드러운 애니메이션 효과를 위한 조건부 렌더링 */}
      <div style={{ 
        maxHeight: isAllChecked ? "200px" : "0", 
        overflow: "hidden", 
        transition: "max-height 0.5s ease-in-out" 
      }}>
        <p style={{ color: darkPurple, fontSize: "12px", marginTop: "10px" }}>아래 문장을 똑같이 입력해주세요:</p>
        <p style={{ color: "red", fontWeight: "bold", fontSize: "12px" }}>"{targetText}"</p>
        <input
          style={inputStyle}
          placeholder="문장을 입력하세요"
          value={confirmText}
          onChange={(e) => setConfirmText(e.target.value)}
        />
        <button style={buttonStyle} onClick={handleWithdraw} disabled={confirmText !== targetText}>
          탈퇴 확정
        </button>
      </div>

      {message.text && (
        <div style={{ marginTop: "15px", padding: "10px", backgroundColor: message.type === "error" ? "#D9534F" : "#5CB85C", color: "white", borderRadius: "10px" }}>
          {message.text}
        </div>
      )}
    </div>
  );
};

export default WithdrawalPage;