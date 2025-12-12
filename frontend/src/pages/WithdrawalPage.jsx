import React, { useState, useEffect } from "react";
import apiClient from "../api/apiClient";
import "../theme/theme.css";

const WithdrawalPage = ({ onLogout }) => {
  const [checks, setChecks] = useState({ info: false, email: false, law: false });
  const [confirmText, setConfirmText] = useState("");
  const [message, setMessage] = useState({ text: "", type: "" });
  const isAllChecked = checks.info && checks.email && checks.law;
  const [error, setError] = useState("");
  const [form, setForm] = useState({
    customer_id: "",
  });
  useEffect(() => {
    const fetchId = async () => {
      setError("");
      try {
        const res = await apiClient.get("/api/profile/account");
        const data = res.data; // AccountInfoDto

        setForm({
          customer_id: data.id || "", // 아이디만 가져와
        });
      } catch (err) {
        console.error("아이디 조회 실패:", err);
        setError(
          err.response?.data?.message ||
          "회원 정보를 불러오는 중 오류가 발생했습니다."
        );
      }
    };
    fetchId();
  }, []);
  const handleCheck = (e) => {
    const { name, checked } = e.target;
    setChecks((prev) => ({ ...prev, [name]: checked }));
  };
  const targetText = `${form.customer_id} 회원 탈퇴를 진행하는 것에 동의합니다.`;
  const handleWithdraw = async () => {
    if (confirmText !== targetText) {
      setMessage({ text: "입력된 문장이 일치하지 않습니다.", type: "error" });
      return;
    }

    try {
      await apiClient.delete(`/api/auth/withdrawal`);
      setMessage({ text: "탈퇴 처리가 완료되었습니다. 복구 코드가 이메일로 발송되었습니다.", type: "success" });
      setTimeout(() => onLogout(), 2000); // App.jsx의 handleLogoutSuccess 호출
    } catch (error) {
      const errorMessage = error.response?.data?.message || "탈퇴 처리 중 알 수 없는 오류가 발생했습니다.";
      console.error("Withdrawal API Error:", error);
      setMessage({ text: errorMessage, type: "error" });
    }
  };

  const darkPurple = "#78266A";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";

  const containerStyle = {
    display: "flex",
    justifyContent: "center", // 가로 중앙 정렬
    alignItems: "center", // 세로 중앙 정렬
    width: "100vw", // 전체 뷰포트 너비
    height: "100vh", // 전체 뷰포트 높이
  };

  const boxStyle = {
    padding: "30px 40px",
    width: "400px",
    maxWidth: "90%",
    borderRadius: "15px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)",
    textAlign: "center",
    backgroundColor: lightPeach,
  };

  const checkGroupStyle = {
    textAlign: "left",
    marginBottom: "15px",
    color: darkPurple,
    fontSize: "15px",
    lineHeight: "2",
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
    <div style={containerStyle}><div style={boxStyle}>
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
    </div></div>
  );
};

export default WithdrawalPage;