import React, { useState, useCallback } from "react";
import axios from "axios";
import "../theme/theme.css"; 

const baseURL = import.meta.env.VITE_LOCAL_BASE_URL;

const SignupPage = ({ onToggleMode, onSignupSuccess }) => {
  // ------------------------------------
  // 1. 상태 관리
  // ------------------------------------
  const [customer_id, setcustomer_id] = useState("");
  const [idChecked, setIdChecked] = useState(false); // ID 중복 확인 여부

  const [email, setEmail] = useState("");
  const [emailChecked, setEmailChecked] = useState(false); // Email 중복 확인 여부

  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [birthDate, setBirthDate] = useState("");
  const [age, setAge] = useState(0);

  // 메시지 표시 상태
  const [message, setMessage] = useState({ text: "", type: "" });

  // ------------------------------------
  // 2. 상태 설정 및 계산 로직 (기존 유지)
  // ------------------------------------
  const createHandleChange = useCallback(
    (setter) => (e) => {
      setter(e.target.value);
    },
    []
  );

  const calculateAge = (dobString) => {
    if (!dobString) return 0;
    const today = new Date();
    const birthDate = new Date(dobString);
    if (isNaN(birthDate)) return 0;

    let calculatedAge = today.getFullYear() - birthDate.getFullYear();
    const monthDifference = today.getMonth() - birthDate.getMonth();

    if (
      monthDifference < 0 ||
      (monthDifference === 0 && today.getDate() < birthDate.getDate())
    ) {
      calculatedAge--;
    }
    return calculatedAge > 0 ? calculatedAge : 0;
  };

  const handleBirthDateChange = useCallback((e) => {
    const newDate = e.target.value;
    setBirthDate(newDate);
    const calculatedAge = calculateAge(newDate);
    setAge(calculatedAge);
  }, []);

  // ------------------------------------
  // 3. 중복 확인 핸들러 (추가)
  // ------------------------------------
  const checkIdDuplicate = async () => {
    if (!customer_id) {
      setMessage({ text: "아이디를 입력해주세요.", type: "error" });
      return;
    }
    try {
      const res = await axios.get(`${baseURL}/api/auth/check-id`, { params: { id: customer_id } });
      if (res.data.available) {
        setMessage({ text: "사용 가능한 아이디입니다.", type: "success" });
        setIdChecked(true);
      } else {
        setMessage({ text: "이미 사용 중인 아이디입니다.", type: "error" });
        setIdChecked(false);
      }
    } catch (error) {
      setMessage({ text: "서버 통신 중 오류가 발생했습니다.", type: "error" });
    }
  };

  const checkEmailDuplicate = async () => {
    if (!email) {
      setMessage({ text: "이메일을 입력해주세요.", type: "error" });
      return;
    }
    try {
      const res = await axios.get(`${baseURL}/api/auth/check-email`, { params: { email: email } });
      if (res.data.available) {
        setMessage({ text: "사용 가능한 이메일입니다.", type: "success" });
        setEmailChecked(true);
      } else {
        setMessage({ text: "이미 사용 중인 이메일입니다.", type: "error" });
        setEmailChecked(false);
      }
    } catch (error) {
      setMessage({ text: "서버 통신 중 오류가 발생했습니다.", type: "error" });
    }
  };

  // ------------------------------------
  // 4. 회원가입 실행 (로직 강화)
  // ------------------------------------
  const handleRegister = async () => {
    // 중복 확인 여부 검증
    if (!idChecked || !emailChecked) {
      setMessage({ text: "아이디와 이메일 중복 확인을 완료해주세요.", type: "error" });
      return;
    }

    if (!customer_id || !password || !confirmPassword || !email || !birthDate) {
      setMessage({ text: "모든 필드를 입력해주세요.", type: "error" });
      return;
    }

    if (password !== confirmPassword) {
      setMessage({ text: "비밀번호 확인이 일치하지 않습니다.", type: "error" });
      return;
    }

    const registerData = {
      id: customer_id,
      password: password,
      email: email,
      birth: birthDate,
      age: age,
    };

    // 로딩 대기 없이 즉시 이메일 인증 페이지로 전환
    onSignupSuccess(email);

    try {
      // 가입 요청은 백그라운드에서 실행
      await axios.post(`${baseURL}/api/auth/signup`, registerData, { withCredentials: true });
    } catch (error) {
      console.error("회원가입 비동기 요청 실패:", error);
    }
  };

  // ------------------------------------
  // 5. 스타일 정의 (기존 스타일 엄격 유지)
  // ------------------------------------
  const darkPurple = "#78266A";
  const deppDarkPurple = "#5B2C6F";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";
  const customFont = "PartialSans,SchoolSafetyRoundedSmile,sans-serif";
  const clearCustomFont = "SchoolSafetyRoundedSmile,sans-serif";

  const fontFaceCss = `@font-face { font-family: 'PartialSans'; src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2307-1@1.1/PartialSansKR-Regular.woff2') format('woff2'); font-weight: normal; font-display: swap; }`;
  const fontClearCss = `@font-face { font-family: 'SchoolSafetyRoundedSmile'; src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-5@1.0/HakgyoansimDunggeunmisoTTF-R.woff2') format('woff2'); font-weight: normal; font-display: swap; } @font-face { font-family: 'SchoolSafetyRoundedSmile'; src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-5@1.0/HakgyoansimDunggeunmisoTTF-B.woff2') format('woff2'); font-weight: 700; font-display: swap; }`;
  const fontSet = [fontClearCss, fontFaceCss];

  const textShadowStyle = { textShadow: `4px 4px 2px ${darkPurple}` };
  const textRoundStyle = { textShadow: `-2px 0 ${darkPurple}, 0 2px ${darkPurple}, 2px 0 ${darkPurple}, 0 -2px ${darkPurple}` };
  const containerStyle = { display: "flex", justifyContent: "center", alignItems: "center", width: "100vw", height: "100vh", backgroundColor: darkPurple, fontFamily: customFont };
  const loginBoxStyle = { backgroundColor: lightPeach, padding: "60px 40px", borderRadius: "40px", boxShadow: "0 4px 15px rgba(0, 0, 0, 0.3)", width: "45vh", textAlign: "center", fontFamily: customFont };
  const inputGroupStyle = { marginBottom: "20px", textAlign: "left" };
  const labelStyle = { fontSize: "20px", fontWeight: "0", color: white, marginBottom: "5px", display: "block", letterSpacing: "2px", ...textRoundStyle };
  const titleStyle = { fontSize: "32px", fontWeight: "100", color: white, margin: "25px", display: "block", letterSpacing: "2px", ...textShadowStyle };
  const inputStyle = { width: "100%", padding: "12px 10px", margin: "8px 0", border: "none", borderRadius: "20px", boxSizing: "border-box", outline: "none", fontSize: "16px", backgroundColor: white, color: darkPurple, fontFamily: clearCustomFont, fontWeight: 700, boxShadow: `4px 4px 0px ${darkPurple}` };
  const buttonStyle = { backgroundColor: white, color: darkPurple, padding: "10px 30px", fontSize: "15px", fontWeight: "100", borderRadius: "20px", border: `2px solid ${darkPurple}`, cursor: "pointer", marginTop: "10px", margin: "5px", transition: "background-color 0.3s", fontFamily: customFont, boxShadow: `4px 4px 0px ${darkPurple}` };

  const messageStyle = { padding: '10px', borderRadius: '10px', color: white, fontFamily: clearCustomFont, backgroundColor: message.type === 'error' ? '#D9534F' : '#5CB85C', fontSize: '14px', marginBottom: '15px' };

  // ------------------------------------
  // 6. 렌더링
  // ------------------------------------
  return (
    <div style={containerStyle}>
      <style>{fontSet}</style>
      <div style={loginBoxStyle}>
        <div><h2 style={titleStyle}>회원가입</h2></div>

        {/* 메시지 영역 */}
        {message.text && <div style={messageStyle}>{message.text}</div>}

        {/* 1. ID 입력 (버튼 포함) */}
        <div style={inputGroupStyle}>
          <label style={labelStyle}>ID</label>
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            <input 
              type="text" 
              style={{ ...inputStyle, flex: 1, margin: 0 }} 
              value={customer_id} 
              onChange={(e) => { setcustomer_id(e.target.value); setIdChecked(false); setMessage({text:"", type:""}); }} 
              placeholder="아이디" 
            />
            <button type="button" style={{ ...buttonStyle, marginTop: 0, fontSize: "11px", padding: "8px 12px", whiteSpace: "nowrap" }} onClick={checkIdDuplicate}>
              {idChecked ? "확인됨" : "중복확인"}
            </button>
          </div>
        </div>

        {/* 2. Email 입력 (버튼 포함) */}
        <div style={inputGroupStyle}>
          <label style={labelStyle}>Email</label>
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            <input 
              type="email" 
              style={{ ...inputStyle, flex: 1, margin: 0 }} 
              value={email} 
              onChange={(e) => { setEmail(e.target.value); setEmailChecked(false); setMessage({text:"", type:""}); }} 
              placeholder="이메일" 
            />
            <button type="button" style={{ ...buttonStyle, marginTop: 0, fontSize: "11px", padding: "8px 12px", whiteSpace: "nowrap" }} onClick={checkEmailDuplicate}>
              {emailChecked ? "확인됨" : "중복확인"}
            </button>
          </div>
        </div>

        {/* 3. 비밀번호 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_password" style={labelStyle}>비밀번호</label>
          <input type="password" id="reg_password" placeholder="비밀번호" style={inputStyle} value={password} onChange={createHandleChange(setPassword)} />
        </div>

        {/* 4. 비밀번호 확인 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_confirmPassword" style={labelStyle}>비밀번호 확인</label>
          <input type="password" id="reg_confirmPassword" placeholder="비밀번호 재입력" style={inputStyle} value={confirmPassword} onChange={createHandleChange(setConfirmPassword)} />
        </div>

        {/* 5. 생년월일 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_birthDate" style={labelStyle}>생년월일</label>
          <input type="date" id="reg_birthDate" style={inputStyle} value={birthDate} onChange={handleBirthDateChange} />
          <div style={{ color: darkPurple, marginTop: "5px" }}>
            계산된 나이: <span style={{ fontWeight: "bold" }}>{age} 세</span>
          </div>
        </div>

        <div>
          <button type="button" style={buttonStyle} onClick={handleRegister}>회원가입 및 이메일 인증</button>
          <button type="button" style={buttonStyle} onClick={onToggleMode}>로그인 페이지로</button>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;