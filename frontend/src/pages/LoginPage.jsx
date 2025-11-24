import React, { useState } from "react"; // useState 훅을 사용하기 위해 import

const LoginPage = () => {
  // ------------------------------------
  // 1. 상태 관리 (State Management)
  // ------------------------------------
  const [customerId, setCustomerId] = useState(""); // ID 입력 상태
  const [password, setPassword] = useState(""); // PW 입력 상태

  // ------------------------------------
  // 2. 로그인 처리 함수 (Login Handler)
  // ------------------------------------
  const handleLogin = async () => {
    // 폼 제출 방지 (버튼의 기본 동작 방지)
    // <button>에 type="submit"이 아닌 type="button"을 사용하거나, onClick에서 event.preventDefault()를 사용합니다.

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
          // 백엔드 요구 사항에 맞게 필드명 사용
          id: customerId,
          password: password,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("로그인 성공:", data);
        alert("로그인에 성공했습니다! 메인 페이지로 이동합니다.");
        // TODO: 로그인 성공 후 토큰 저장 및 페이지 이동 로직 구현 (예: navigate('/main'))
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
  // 3. 스타일 및 변수 정의
  // ------------------------------------

  // 폰트 정의를 위한 CSS 문자열
  const fontFaceCss = `
    @font-face {
      font-family: 'PartialSans';
      src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2307-1@1.1/PartialSansKR-Regular.woff2') format('woff2');
      font-weight: normal;
      font-display: swap;
    }
  `;

  // 이미지에서 사용된 색상 변수
  const darkPurple = "#5B2C6F";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";
  const brown = "#4E2A1A";
  const customFont = "PartialSans, sans-serif";

  // 텍스트 그림자 스타일 정의
  const textShadowStyle = {
    textShadow: `4px 4px 2px ${darkPurple}`,
  };

  // 컴포넌트 스타일 정의
  const containerStyle = {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    width: "100vw", // 뷰포트 전체 너비
    height: "100vh", // 뷰포트 전체 높이
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
    //marginBottom: "5px",
    maxWidth: "100%",
    height: "auto",
  };

  const logoTextStyle = {
    fontSize: "24px",
    fontWeight: "bold",
    color: brown,
    marginBottom: "10px",
    ...textShadowStyle,
  };

  const iconStyle = {
    fontSize: "40px",
    color: brown,
    ...textShadowStyle,
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
    marginTop: "30px",
    transition: "background-color 0.3s",
    fontFamily: customFont,
    boxShadow: `4px 4px 0px ${darkPurple}`,
  };

  // ------------------------------------
  // 4. 컴포넌트 렌더링 (Component Render)
  // ------------------------------------

  return (
    <div style={containerStyle}>
      {/* 폰트 로딩을 위한 <style> 태그를 동적으로 삽입합니다. */}
      <style>{fontFaceCss}</style>

      <div style={loginBoxStyle}>
        {/* 로고 영역 */}
        <div >
          {/* alt 속성을 "DISH 로고"로 수정했습니다. */}
          <img
            src="..\src\assets\DISH_LOGO.png"
            alt="DISH 로고"
            style={logoContainerStyle}
          />
        </div>

        {/* ID 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="customerId" style={labelStyle}>
            ID
          </label>
          <input
            type="text"
            id="customerId"
            placeholder="아이디를 입력하세요"
            style={inputStyle}
            value={customerId} // **상태 값 연결**
            onChange={(e) => setCustomerId(e.target.value)} // **변경 핸들러 연결**
          />
        </div>

        {/* PW 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="password" style={labelStyle}>
            PW
          </label>{" "}
          {/* htmlFor를 "password"로 변경하여 input id와 일치시킴 */}
          <input
            type="password"
            id="password" // **id를 password로 변경하여 상태 및 label과 일관성 유지**
            placeholder="비밀번호를 입력하세요"
            style={inputStyle}
            value={password} // **상태 값 연결**
            onChange={(e) => setPassword(e.target.value)} // **변경 핸들러 연결**
          />
        </div>

        {/* 로그인 버튼 */}
        <div>
          <button type="button" style={buttonStyle} onClick={handleLogin}>
            로그인
          </button>
        </div>

        <div>
          {/* 회원가입 버튼 */}
          <button type="button" style={buttonStyle} onClick={handleLogin}>
            회원가입
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
