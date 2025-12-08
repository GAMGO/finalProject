// src/pages/SignupPage.jsx
import React, { useState, useCallback, useEffect } from "react";
import axios from "axios";
import "../theme/theme.css"; // ✅ 테마 CSS

const baseURL = import.meta.env.VITE_LOCAL_BASE_URL;

const SignupPage = ({ onToggleMode, onSignupSuccess }) => {
  // ------------------------------------
  // 1. 상태 관리
  // ------------------------------------
  const [customer_id, setcustomer_id] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [email, setEmail] = useState("");
  const [birthDate, setBirthDate] = useState("");
  const [age, setAge] = useState(0);

  // ✅ 로그인/회원가입 공통 테마 상태 (light | dark)
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "light";
    return localStorage.getItem("theme") || "light";
  });

  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-theme", theme);
    }
    if (typeof window !== "undefined") {
      localStorage.setItem("theme", theme);
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const isDark = theme === "dark";

  // ------------------------------------
  // 2. 상태 설정 함수
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

  const handleRegister = async () => {
    if (!customer_id || !password || !confirmPassword || !email || !birthDate) {
      alert("모든 필드를 입력해주세요.");
      return;
    }
    if (password !== confirmPassword) {
      alert("비밀번호 확인이 일치하지 않습니다.");
      return;
    }

    const registerData = {
      id: customer_id,
      password: password,
      email: email,
      birth: birthDate,
      age: age,
    };

    try {
      const response = await axios.post(
        `${baseURL}/api/auth/signup`,
        registerData,
        { withCredentials: true }
      );

      alert(
        "회원가입이 성공적으로 완료되었습니다! 이메일로 전송된 인증 코드를 입력해주세요."
      );
      onSignupSuccess(email);
    } catch (error) {
      if (error.response) {
        alert(
          `회원가입 실패: ${
            error.response.data.message || "이미 존재하는 사용자 이름입니다."
          }`
        );
        console.error("회원가입 에러 응답:", error.response);
      } else if (error.request) {
        alert(
          "서버 응답이 없습니다. CORS 설정 또는 네트워크 상태를 확인해주세요."
        );
        console.error("회원가입 에러 요청:", error.request);
      } else {
        alert("서버 연결에 실패했습니다. 네트워크 상태를 확인해주세요.");
        console.error("회원가입 에러:", error.message);
      }
    }
  };

  // ------------------------------------
  // 4. 스타일 정의
  // ------------------------------------
  const darkPurple = "#78266A";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";
  const logoDarkBrown = "#5e3a31";
  const chocolateShadow = "#3a221c";

  const customFont = "PartialSans,SchoolSafetyRoundedSmile,sans-serif";
  const clearCustomFont = "SchoolSafetyRoundedSmile,sans-serif";

  const fontFaceCss = `
    @font-face {
      font-family: 'PartialSans';
      src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2307-1@1.1/PartialSansKR-Regular.woff2') format('woff2');
      font-weight: normal;
      font-display: swap;
    }
  `;
  const fontClearCss = `
@font-face {
  font-family: 'SchoolSafetyRoundedSmile';
  src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-5@1.0/HakgyoansimDunggeunmisoTTF-R.woff2') format('woff2');
  font-weight: normal;
  font-display: swap;
}
@font-face {
  font-family: 'SchoolSafetyRoundedSmile';
  src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-5@1.0/HakgyoansimDunggeunmisoTTF-B.woff2') format('woff2');
  font-weight: 700;
  font-display: swap;
}`;
  const fontSet = [fontClearCss, fontFaceCss];

  const textShadowStyle = { textShadow: `4px 4px 2px ${darkPurple}` };
  const textRoundStyle = {
    textShadow: `-2px 0 ${darkPurple}, 0 2px ${darkPurple}, 2px 0 ${darkPurple}, 0 -2px ${darkPurple}`,
  };

  const containerStyle = {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    width: "100vw",
    height: "100vh",
    backgroundColor: isDark ? "#4A1010" : darkPurple,
    fontFamily: customFont,
    color: isDark ? "#fef3e8" : "#222222",
  };

  const loginBoxStyle = {
    backgroundColor: isDark ? logoDarkBrown : lightPeach,
    padding: "60px 40px",
    borderRadius: "40px",
    boxShadow: isDark
      ? "0 10px 30px rgba(0, 0, 0, 0.7)"
      : "0 4px 15px rgba(0, 0, 0, 0.3)",
    width: "45vh",
    textAlign: "center",
    fontFamily: customFont,
  };

  const inputGroupStyle = { marginBottom: "20px", textAlign: "left" };

  const labelStyle = {
    fontSize: "20px",
    fontWeight: "0",
    color: white,
    marginBottom: "5px",
    display: "block",
    letterSpacing: "2px",
    ...textRoundStyle,
  };

  const titleStyle = {
    fontSize: "32px",
    fontWeight: "100",
    color: white,
    margin: "25px",
    display: "block",
    letterSpacing: "2px",
    ...textShadowStyle,
  };

  // ✅ 다크 모드: 입력칸 연한 갈색, 글자는 진한 갈색
  const inputStyle = {
    width: "100%",
    padding: "12px 10px",
    margin: "8px 0",
    border: "none",
    borderRadius: "20px",
    boxSizing: "border-box",
    outline: "none",
    fontSize: "16px",
    backgroundColor: isDark ? lightPeach : white,
    color: isDark ? logoDarkBrown : darkPurple,
    fontFamily: clearCustomFont,
    fontWeight: 700,
    boxShadow: `4px 4px 0px ${
      isDark ? chocolateShadow : darkPurple
    }`,
  };

  const buttonStyle = {
    backgroundColor: isDark ? lightPeach : white,
    color: isDark ? logoDarkBrown : darkPurple,
    padding: "10px 30px",
    fontSize: "15px",
    fontWeight: "100",
    borderRadius: "20px",
    border: `2px solid ${isDark ? chocolateShadow : darkPurple}`,
    cursor: "pointer",
    marginTop: "10px",
    margin: "5px",
    transition: "background-color 0.3s",
    fontFamily: customFont,
    boxShadow: `4px 4px 0px ${
      isDark ? chocolateShadow : darkPurple
    }`,
  };

  // ------------------------------------
  // 5. 컴포넌트 렌더링
  // ------------------------------------
  return (
    <div style={containerStyle}>
      <style>{fontSet}</style>

      {/* ✅ 회원가입 화면용 라이트/다크 토글 */}
      <button
        type="button"
        className="side-nav-btn theme-toggle-btn auth-theme-toggle"
        onClick={toggleTheme}
      >
        {theme === "light" ? "다크 모드" : "라이트 모드"}
      </button>

      <div style={loginBoxStyle}>
        <div>
          <h2 style={titleStyle}>회원가입</h2>
        </div>

        {/* 1. ID 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_customer_id" style={labelStyle}>
            ID
          </label>
          <input
            type="text"
            id="reg_customer_id"
            placeholder="사용할 아이디를 입력하세요"
            style={inputStyle}
            value={customer_id}
            onChange={createHandleChange(setcustomer_id)}
          />
        </div>

        {/* 2. Email 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_email" style={labelStyle}>
            Email
          </label>
          <input
            type="email"
            id="reg_email"
            placeholder="이메일 주소를 입력하세요"
            style={inputStyle}
            value={email}
            onChange={createHandleChange(setEmail)}
          />
        </div>

        {/* 3. 비밀번호 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_password" style={labelStyle}>
            비밀번호
          </label>
          <input
            type="password"
            id="reg_password"
            placeholder="비밀번호를 입력하세요"
            style={inputStyle}
            value={password}
            onChange={createHandleChange(setPassword)}
          />
        </div>

        {/* 4. 비밀번호 확인 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_confirmPassword" style={labelStyle}>
            비밀번호 확인
          </label>
          <input
            type="password"
            id="reg_confirmPassword"
            placeholder="비밀번호를 다시 입력하세요"
            style={inputStyle}
            value={confirmPassword}
            onChange={createHandleChange(setConfirmPassword)}
          />
        </div>

        {/* 5. 생년월일 + 나이 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_birthDate" style={labelStyle}>
            생년월일
          </label>
          <input
            type="date"
            id="reg_birthDate"
            style={inputStyle}
            value={birthDate}
            onChange={handleBirthDateChange}
          />
          <div style={{ color: isDark ? lightPeach : darkPurple, marginTop: 6 }}>
            계정당 나이: <span>{age} 세</span>
          </div>
        </div>

        <div>
          <button type="button" style={buttonStyle} onClick={handleRegister}>
            회원가입 완료 및 이메일 인증하기
          </button>

          <button type="button" style={buttonStyle} onClick={onToggleMode}>
            이미 계정이 있으신가요? 로그인 페이지로
          </button>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;
