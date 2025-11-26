import React, { useState } from "react";

const LoginPage = () => {
  // ------------------------------------
  // 1. 상태 관리 (State Management)
  // ------------------------------------
  const [customerId, setCustomerId] = useState(""); // ID 입력 상태
  const [password, setPassword] = useState(""); // PW 입력 상태
  const [isLoginMode, setIsLoginMode] = useState(true); // 🌟 모드 상태: true=로그인, false=회원가입
  
  // 🌟 회원가입 모드에서 필요한 추가 상태
  const [confirmPassword, setConfirmPassword] = useState(""); // 비밀번호 확인
  const [email, setEmail] = useState(""); // 이메일

  // ------------------------------------
  // 2. 로그인 처리 함수 (Login Handler)
  // ------------------------------------
  const handleLogin = async () => {
    if (!customerId || !password) {
      alert("아이디와 비밀번호를 모두 입력해주세요.");
      return;
    }

    try {
      // 실제 백엔드 로그인 API 엔드포인트로 변경해야 합니다.
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          id: customerId,
          password: password,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("로그인 성공:", data);
        alert("로그인에 성공했습니다! 메인 페이지로 이동합니다.");
        // TODO: 로그인 성공 후 토큰 저장 및 페이지 이동 로직 구현
      } else {
        const errorData = await response.json();
        console.error("로그인 실패 응답:", errorData);
        alert(
          `로그인 실패: ${
            errorData.message || "아이디 또는 비밀번호를 다시 확인해주세요."
          }`
        );
      }
    } catch (error) {
      console.error("서버 연결 오류:", error);
      alert("서버 연결에 실패했습니다. 네트워크 상태를 확인해주세요.");
    }
  };

  // ------------------------------------
  // 2. 회원가입 처리 함수 (Register Handler)
  // ------------------------------------
  const handleRegister = async () => {
    if (!customerId || !password || !confirmPassword || !email) {
      alert("모든 필드를 입력해주세요.");
      return;
    }

    if (password !== confirmPassword) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }

    // 실제 백엔드 회원가입 API 엔드포인트로 변경해야 합니다.
    try {
      const response = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          id: customerId, 
          password: password, 
          email: email 
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("회원가입 성공:", data);
        alert("회원가입에 성공했습니다! 로그인 페이지로 이동합니다.");
        
        // 성공 후 로그인 모드로 전환 및 폼 초기화
        setIsLoginMode(true); 
        setCustomerId("");
        setPassword("");
        setConfirmPassword("");
        setEmail("");

      } else {
        const errorData = await response.json();
        console.error("회원가입 실패 응답:", errorData);
        alert(
          `회원가입 실패: ${
            errorData.message || "이미 존재하는 아이디이거나 서버 오류입니다."
          }`
        );
      }
    } catch (error) {
      console.error("서버 연결 오류:", error);
      alert("서버 연결에 실패했습니다. 네트워크 상태를 확인해주세요.");
    }
  };
  
  // ------------------------------------
  // 3. 스타일 및 변수 정의
  // ------------------------------------

  const fontFaceCss = `
    @font-face {
      font-family: 'PartialSans';
      src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2307-1@1.1/PartialSansKR-Regular.woff2') format('woff2');
      font-weight: normal;
      font-display: swap;
    }
  `;

  const darkPurple = "#5B2C6F";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";
  const customFont = "PartialSans, sans-serif";

  const textShadowStyle = {
    textShadow: `4px 4px 2px ${darkPurple}`,
  };

  const containerStyle = {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    width: "100vw",
    height: "100vh",
    backgroundColor: darkPurple,
    fontFamily: customFont,
  };

  const loginBoxStyle = {
    backgroundColor: lightPeach,
    padding: "60px 40px",
    borderRadius: "40px",
    boxShadow: "0 4px 15px rgba(0, 0, 0, 0.3)",
    width: "45vh",
    textAlign: "center",
    fontFamily: customFont,
  };

  const logoContainerStyle = {
    maxWidth: "100%",
    height: "auto",
  };

  const inputGroupStyle = {
    marginBottom: "20px",
    textAlign: "left",
  };

  const labelStyle = {
    fontSize: "18px",
    fontWeight: "bold",
    color: white,
    marginBottom: "5px",
    display: "block",
    letterSpacing: "2px",
    ...textShadowStyle,
  };

  const inputStyle = {
    width: "100%",
    padding: "12px 10px",
    margin: "8px 0",
    border: "none",
    borderRadius: "20px",
    boxSizing: "border-box",
    outline: "none",
    fontSize: "16px",
    backgroundColor: white,
    color: darkPurple,
    fontFamily: customFont,
    boxShadow: `4px 4px 0px ${darkPurple}`,
  };

  const buttonStyle = {
    backgroundColor: white,
    color: darkPurple,
    padding: "10px 30px",
    fontSize: "18px",
    fontWeight: "bold",
    borderRadius: "20px",
    border: `2px solid ${darkPurple}`,
    cursor: "pointer",
    marginTop: "20px",
    margin: "10px", 
    transition: "background-color 0.3s",
    fontFamily: customFont,
    boxShadow: `4px 4px 0px ${darkPurple}`,
  };

  // ------------------------------------
  // 4. 서브 컴포넌트 (폼) 정의
  // ------------------------------------
  
  // 🌟 로그인 폼
  const LoginForm = () => (
    <>
      <div style={inputGroupStyle}>
        <label htmlFor="customerId" style={labelStyle}>
          ID
        </label>
        <input
          type="text"
          id="customerId"
          placeholder="아이디를 입력하세요"
          style={inputStyle}
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
        />
      </div>

      <div style={inputGroupStyle}>
        <label htmlFor="password" style={labelStyle}>
          PW
        </label>
        <input
          type="password"
          id="password"
          placeholder="비밀번호를 입력하세요"
          style={inputStyle}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      </div>

      <div>
        <button type="button" style={buttonStyle} onClick={handleLogin}>
          로그인
        </button>
      </div>

      <div>
        <button 
          type="button" 
          style={buttonStyle} 
          onClick={() => {
            setIsLoginMode(false); // 회원가입 모드로 전환
            setCustomerId("");
            setPassword("");
          }}
        >
          회원가입
        </button>
      </div>
    </>
  );

  // 🌟 회원가입 폼
  const RegisterForm = () => (
    <>
      {/* 1. ID 입력 */}
      <div style={inputGroupStyle}>
        <label htmlFor="reg_customerId" style={labelStyle}>
          ID
        </label>
        <input
          type="text"
          id="reg_customerId"
          placeholder="사용할 아이디를 입력하세요"
          style={inputStyle}
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
        />
      </div>

      {/* 2. Email 입력 */}
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
          onChange={(e) => setEmail(e.target.value)}
        />
      </div>

      {/* 3. 비밀번호 입력 */}
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
          onChange={(e) => setPassword(e.target.value)}
        />
      </div>
      
      {/* 4. 비밀번호 확인 입력 */}
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
          onChange={(e) => setConfirmPassword(e.target.value)}
        />
      </div>

      <div>
        <button type="button" style={buttonStyle} onClick={handleRegister}>
          회원가입 완료
        </button>
      </div>
      
      <div>
        <button 
          type="button" 
          style={buttonStyle} 
          onClick={() => {
            setIsLoginMode(true); // 로그인 모드로 전환
            // 폼 필드 초기화
            setCustomerId("");
            setPassword("");
            setConfirmPassword("");
            setEmail("");
          }}
        >
          로그인 페이지로
        </button>
      </div>
    </>
  );
  
  // ------------------------------------
  // 5. 컴포넌트 렌더링 (Component Render)
  // ------------------------------------

  return (
    <div style={containerStyle}>
      {/* 폰트 로딩을 위한 <style> 태그 */}
      <style>{fontFaceCss}</style>

      <div style={loginBoxStyle}>
        {/* 로고 영역 */}
        <div>
          <img
            src="..\src\assets\DISH_LOGO.png"
            alt="DISH 로고"
            style={logoContainerStyle}
          />
        </div>
        
        {/* 🌟 조건부 렌더링: 모드에 따라 폼 전환 */}
        {isLoginMode ? <LoginForm /> : <RegisterForm />}

      </div>
    </div>
  );
};

export default LoginPage;