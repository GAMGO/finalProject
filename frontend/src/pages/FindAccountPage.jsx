import React, { useState } from "react";
import axios from "axios";
import "../theme/theme.css";

const baseURL = import.meta.env.VITE_LOCAL_BASE_URL;

// 이 페이지는 FindAccountController의 /api/account/* 엔드포인트를 사용합니다.
const FindAccountPage = ({ onGoToLogin, initialMode = 'findId' }) => { // ✅ initialMode 추가
  // ------------------------------------
  // 1. 상태 관리
  // ------------------------------------
  const [mode, setMode] = useState(initialMode); // 'findId' 또는 'findPassword'
  const [form, setForm] = useState({
      id: "",
      email: "",
      birthDate: "" // YYYY-MM-DD
  });
  const [message, setMessage] = useState({ text: "", type: "" });
  const [step, setStep] = useState(1); // PW 찾기 전용: 1: 정보 확인, 2: 코드/새 비번 입력
  const [recoveryCode, setRecoveryCode] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmNewPassword, setConfirmNewPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [foundId, setFoundId] = useState(""); // 찾은 아이디 저장
  
  // ------------------------------------
  // 2. 스타일 정의 (유지)
  // ------------------------------------
  const darkPurple = "#78266A";
  const lightPeach = "#F5D7B7";
  const customFont = "PartialSans, SchoolSafetyRoundedSmile, sans-serif";
  
  const containerStyle = {
    display: "flex", justifyConent: "center", alignItems: "center", width: "100vw", height: "100vh", backgroundColor: darkPurple, fontFamily: customFont,
  };
  const boxStyle = { backgroundColor: lightPeach, padding: "40px 40px", borderRadius: "20px", width: "45vh", textAlign: "center" };
  const inputStyle = { width: "100%", padding: "12px", borderRadius: "10px", border: "2px solid #ccc", marginBottom: "10px", fontSize: "16px" };
  const buttonStyle = { 
    width: "100%", backgroundColor: darkPurple, color: "#FFFFFF", padding: "10px 20px", borderRadius: "10px", border: "none", marginTop: "15px", cursor: "pointer", fontWeight: "bold" 
  };
  const inputGroupStyle = { marginBottom: "15px", textAlign: "left" };
  const labelStyle = { display: "block", marginBottom: "5px", color: darkPurple, fontWeight: "bold" };

  // ------------------------------------
  // 3. 아이디 찾기 핸들러
  // ------------------------------------
  const handleFindId = async (e) => {
    e.preventDefault();
    setMessage({ text: "", type: "" });
    setLoading(true);

    const { email, birthDate } = form;
    if (!email || !birthDate) {
        setMessage({ text: "이메일과 생년월일을 입력해주세요.", type: "error" });
        setLoading(false);
        return;
    }

    try {
        // 🚨 엔드포인트 사용: POST /api/account/id/find
        const response = await axios.post(`${baseURL}/api/account/id/find`, {
            email,
            birthDate
        });
        
        setFoundId(response.data.id);
        setMessage({
            text: `찾으신 아이디는 '${response.data.id}' 입니다.`,
            type: "success",
        });

    } catch (error) {
        const errorMessage = error.response?.data?.message || "입력 정보와 일치하는 계정을 찾을 수 없습니다.";
        setMessage({ text: errorMessage, type: "error" });
        setFoundId("");
    } finally {
        setLoading(false);
    }
  };
  
  // ------------------------------------
  // 4. 비밀번호 찾기 핸들러 (Step 1: 정보 확인 및 코드 발송)
  // ------------------------------------
  const handleSendRecoveryCode = async (e) => {
    e.preventDefault();
    setMessage({ text: "", type: "" });
    setLoading(true);

    const { id, email, birthDate } = form;
    if (!id || !email || !birthDate) {
        setMessage({ text: "모든 정보를 입력해주세요.", type: "error" });
        setLoading(false);
        return;
    }

    try {
        // 🚨 엔드포인트 사용: POST /api/account/password/send-code
        await axios.post(`${baseURL}/api/account/password/send-code`, {
            id,
            email,
            birthDate
        });
        
        setMessage({
            text: "인증 코드가 이메일로 발송되었습니다. 코드를 확인해주세요.",
            type: "success",
        });
        setStep(2); // 다음 단계로 이동

    } catch (error) {
        const errorMessage = error.response?.data?.message || "입력 정보와 일치하는 계정을 찾을 수 없습니다.";
        setMessage({ text: errorMessage, type: "error" });
    } finally {
        setLoading(false);
    }
  };
  
  // ------------------------------------
  // 5. 비밀번호 찾기 핸들러 (Step 2: 코드 검증 및 비밀번호 변경)
  // ------------------------------------
  const handleResetPassword = async (e) => {
    e.preventDefault();
    setMessage({ text: "", type: "" });
    setLoading(true);

    // ... (유효성 검사 생략 - 이전 코드 참고) ...
    if (!recoveryCode || recoveryCode.length !== 6) {
        setMessage({ text: "유효한 6자리 인증 코드를 입력해주세요.", type: "error" });
        setLoading(false);
        return;
    }
    if (newPassword.length < 8) {
        setMessage({ text: "새 비밀번호는 8자 이상이어야 합니다.", type: "error" });
        setLoading(false);
        return;
    }
    if (newPassword !== confirmNewPassword) {
        setMessage({ text: "새 비밀번호와 확인이 일치하지 않습니다.", type: "error" });
        setLoading(false);
        return;
    }
    
    try {
        // 🚨 엔드포인트 사용: POST /api/account/password/reset
        await axios.post(`${baseURL}/api/account/password/reset`, {
            email: form.email, 
            code: recoveryCode,
            newPassword: newPassword,
        });
        
        setMessage({
            text: "비밀번호가 성공적으로 변경되었습니다. 로그인 페이지로 이동합니다.",
            type: "success",
        });
        
        setTimeout(() => {
            if (typeof onGoToLogin === 'function') onGoToLogin();
        }, 3000);

    } catch (error) {
        const errorMessage = error.response?.data?.message || "비밀번호 변경에 실패했습니다. 코드를 다시 확인해주세요.";
        setMessage({ text: errorMessage, type: "error" });
    } finally {
        setLoading(false);
    }
  };
  
  // ------------------------------------
  // 6. 렌더링
  // ------------------------------------
  const resetForm = (newMode) => {
    setMode(newMode);
    setForm({ id: "", email: "", birthDate: "" });
    setMessage({ text: "", type: "" });
    setStep(1);
    setRecoveryCode("");
    setNewPassword("");
    setConfirmNewPassword("");
    setFoundId("");
  };
  
  return (
    <div style={containerStyle}>
      <div style={boxStyle}>
        <h2 style={{ color: darkPurple, marginBottom: "30px" }}>
            {mode === 'findId' ? "아이디 찾기" : "비밀번호 초기화"}
        </h2>
        
        <div style={{ marginBottom: "20px" }}>
            <button 
                onClick={() => resetForm('findId')}
                style={{ ...buttonStyle, width: "48%", marginRight: "4%", 
                         backgroundColor: mode === 'findId' ? darkPurple : lightPeach, 
                         color: mode === 'findId' ? 'white' : darkPurple,
                         border: mode === 'findId' ? 'none' : `1px solid ${darkPurple}` }}
                disabled={loading}
            >
                아이디 찾기
            </button>
            <button 
                onClick={() => resetForm('findPassword')}
                style={{ ...buttonStyle, width: "48%", 
                         backgroundColor: mode === 'findPassword' ? darkPurple : lightPeach, 
                         color: mode === 'findPassword' ? 'white' : darkPurple,
                         border: mode === 'findPassword' ? 'none' : `1px solid ${darkPurple}` }}
                disabled={loading}
            >
                비밀번호 찾기
            </button>
        </div>
        
        {/* 아이디 찾기 UI */}
        {mode === 'findId' && (
            <form onSubmit={handleFindId}>
                <p style={{ color: darkPurple, fontSize: "14px", marginBottom: "20px" }}>
                    가입 시 사용한 이메일과 생년월일을 입력해주세요.
                </p>
                
                <div style={inputGroupStyle}>
                    <label htmlFor="find_id_email" style={labelStyle}>이메일</label>
                    <input
                        type="email"
                        id="find_id_email"
                        placeholder="가입 시 사용한 이메일"
                        style={inputStyle}
                        value={form.email}
                        onChange={(e) => setForm(prev => ({ ...prev, email: e.target.value }))}
                        disabled={loading || foundId}
                    />
                </div>
                <div style={inputGroupStyle}>
                    <label htmlFor="find_id_birthDate" style={labelStyle}>생년월일</label>
                    <input
                        type="date"
                        id="find_id_birthDate"
                        style={inputStyle}
                        value={form.birthDate}
                        onChange={(e) => setForm(prev => ({ ...prev, birthDate: e.target.value }))}
                        disabled={loading || foundId}
                    />
                </div>
                
                {!foundId && (
                    <button 
                        type="submit" 
                        style={{...buttonStyle, opacity: loading ? 0.5 : 1}} 
                        disabled={loading}
                    >
                        {loading ? "찾는 중..." : "아이디 찾기"}
                    </button>
                )}
            </form>
        )}
        
        {/* 비밀번호 초기화 UI */}
        {mode === 'findPassword' && step === 1 && (
            <form onSubmit={handleSendRecoveryCode}>
                <p style={{ color: darkPurple, fontSize: "14px", marginBottom: "20px" }}>
                    계정 확인을 위해 아이디, 이메일, 생년월일을 입력해주세요.
                </p>
                
                <div style={inputGroupStyle}>
                    <label htmlFor="rec_id" style={labelStyle}>아이디</label>
                    <input
                        type="text"
                        id="rec_id"
                        placeholder="아이디"
                        style={inputStyle}
                        value={form.id}
                        onChange={(e) => setForm(prev => ({ ...prev, id: e.target.value }))}
                        disabled={loading}
                    />
                </div>
                <div style={inputGroupStyle}>
                    <label htmlFor="rec_email" style={labelStyle}>이메일</label>
                    <input
                        type="email"
                        id="rec_email"
                        placeholder="가입 시 사용한 이메일"
                        style={inputStyle}
                        value={form.email}
                        onChange={(e) => setForm(prev => ({ ...prev, email: e.target.value }))}
                        disabled={loading}
                    />
                </div>
                <div style={inputGroupStyle}>
                    <label htmlFor="rec_birthDate" style={labelStyle}>생년월일</label>
                    <input
                        type="date"
                        id="rec_birthDate"
                        style={inputStyle}
                        value={form.birthDate}
                        onChange={(e) => setForm(prev => ({ ...prev, birthDate: e.target.value }))}
                        disabled={loading}
                    />
                </div>
                <button 
                    type="submit" 
                    style={{...buttonStyle, opacity: loading ? 0.5 : 1}} 
                    disabled={loading}
                >
                    {loading ? "코드 발송 중..." : "인증 코드 이메일로 받기"}
                </button>
            </form>
        )}
        
        {mode === 'findPassword' && step === 2 && (
            <form onSubmit={handleResetPassword}>
                <p style={{ color: darkPurple, fontSize: "14px", marginBottom: "20px" }}>
                    이메일 ({form.email})로 받은 코드를 입력하고 새 비밀번호를 설정해주세요.
                </p>
                
                {/* 인증 코드 입력 */}
                <div style={inputGroupStyle}>
                    <label htmlFor="rec_code" style={labelStyle}>인증 코드</label>
                    <input
                        type="text"
                        id="rec_code"
                        placeholder="6자리 인증 코드"
                        style={inputStyle}
                        value={recoveryCode}
                        onChange={(e) => setRecoveryCode(e.target.value.replace(/[^0-9]/g, "").slice(0, 6))}
                        maxLength={6}
                        disabled={loading}
                    />
                </div>
                
                {/* 새 비밀번호 입력 */}
                <div style={inputGroupStyle}>
                    <label htmlFor="rec_new_pw" style={labelStyle}>새 비밀번호</label>
                    <input
                        type="password"
                        id="rec_new_pw"
                        placeholder="8자 이상"
                        style={inputStyle}
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        disabled={loading}
                    />
                </div>
                
                {/* 비밀번호 확인 입력 */}
                <div style={inputGroupStyle}>
                    <label htmlFor="rec_confirm_pw" style={labelStyle}>비밀번호 확인</label>
                    <input
                        type="password"
                        id="rec_confirm_pw"
                        placeholder="새 비밀번호 재입력"
                        style={inputStyle}
                        value={confirmNewPassword}
                        onChange={(e) => setConfirmNewPassword(e.target.value)}
                        disabled={loading}
                    />
                </div>
                
                <button 
                    type="submit" 
                    style={{...buttonStyle, opacity: loading ? 0.5 : 1}} 
                    disabled={loading}
                >
                    {loading ? "비밀번호 변경 중..." : "비밀번호 변경 완료"}
                </button>
            </form>
        )}
        
        {/* 메시지 표시 */}
        {message.text && (
            <div 
                style={{ 
                    marginTop: "15px", 
                    color: message.type === "error" ? "red" : darkPurple, 
                    fontWeight: "bold",
                    backgroundColor: message.type === "error" ? "#FFE6E6" : "#E6F3FF",
                    padding: "10px",
                    borderRadius: "10px",
                    fontSize: "14px"
                }}
            >
                {message.text}
            </div>
        )}
        
        {/* 로그인으로 돌아가기 버튼 */}
        <button 
            type="button" 
            style={{ 
                background: "none", 
                border: "none", 
                color: darkPurple, 
                textDecoration: "underline", 
                marginTop: "20px", 
                cursor: "pointer", 
                fontSize: "14px",
                opacity: loading ? 0.5 : 1
            }} 
            onClick={() => { if (!loading) onGoToLogin(); }}
            disabled={loading}
        >
            로그인으로 돌아가기
        </button>
      </div>
    </div>
  );
};

export default FindAccountPage;