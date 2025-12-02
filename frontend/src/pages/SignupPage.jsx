import React, { useState, useCallback } from "react"; // ⭐️ useCallback 추가
import axios from "axios";
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;
const SignupPage = ({ onToggleMode, onSignupSuccess }) => {
  // ------------------------------------
  // 1. 상태 관리
  // ------------------------------------
  const [customer_id, setcustomer_id] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [email, setEmail] = useState("");
  // 생년월일 (YYYY-MM-DD 형식)
  const [birthDate, setBirthDate] = useState("");
  // 성별 (0: 남자, 1: 여자, null/undefined: 선택 안 함)
  const [gender, setGender] = useState('');
  //나이는 자동 계산
  const [age, setAge] = useState(0);

  // ------------------------------------
  // 2. 상태 설정 함수 래핑 (안정성 확보)
  // ------------------------------------

  // 범용 입력 핸들러 함수 (필드별 setter를 호출)
  const createHandleChange = useCallback(
    (setter) => (e) => {
      setter(e.target.value);
    },
    []
  );
  // 생년월일 변경 핸들러: 생년월일과 나이를 동시에 업데이트
  const calculateAge = (dobString) => {
    if (!dobString) return 0;
    const today = new Date();
    const birthDate = new Date(dobString);

    // 날짜 객체가 유효하지 않으면 0 반환
    if (isNaN(birthDate)) return 0;

    let calculatedAge = today.getFullYear() - birthDate.getFullYear();
    const monthDifference = today.getMonth() - birthDate.getMonth();

    // 생일이 지나지 않았으면 1을 뺌 (만 나이 기준)
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

  // 성별 선택 핸들러: 성별 상태(0 또는 1)를 업데이트
  const handleGenderSelect = useCallback((selectedGender) => {
    setGender(selectedGender);
  }, []);

  const handleRegister = async () => {
    if (
      !customer_id ||
      !password ||
      !confirmPassword ||
      !email ||
      !birthDate
    ) {
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
      gender: gender, // 0 또는 1
      age: age, // 자동 계산된 나이
    };

    try {
      const response = await axios.post(
        // ⭐️ 백엔드 회원가입 엔드포인트: https://api.dishinside.shop/api/auth/signup
        `${API_BASE_URL}/api/auth/signup`,
        registerData,
        { withCredentials: true }
      );

      // ⭐️ 회원가입 성공 처리 -> 이메일 인증 바로 진행
      alert("회원가입이 성공적으로 완료되었습니다! 이메일로 전송된 인증 코드를 입력해주세요.");
      onSignupSuccess(email);
    } catch (error) {
      if (error.response) {
        // 서버 응답 (예: 409 Conflict - 이미 존재하는 사용자)
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
  // 4. 스타일 정의 (기존 인라인 스타일 유지)
  // ------------------------------------
  // ... (fontFaceCss, containerStyle 등 모든 스타일 정의 코드는 그대로 유지) ...다른보라색: 5B2C6F
  const darkPurple = "#78266A";
  const deppDarkPurple = "#5B2C6F";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";
  const customFont = "PartialSans, sans-serif";
  const fontFaceCss = `
    @font-face {
      font-family: 'PartialSans';
      src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2307-1@1.1/PartialSansKR-Regular.woff2') format('woff2');
      font-weight: normal;
      font-display: swap;
    }
  `;
  const textShadowStyle = { textShadow: `4px 4px 2px ${darkPurple}` };
  const textRoundStyle = { textShadow: `-2px 0 ${darkPurple}, 0 2px ${darkPurple}, 2px 0 ${darkPurple}, 0 -2px ${darkPurple}` };
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
    fontSize: "15px",
    fontWeight: "100",
    borderRadius: "20px",
    border: `2px solid ${darkPurple}`,
    cursor: "pointer",
    marginTop: "10px",
    margin: "5px",
    transition: "background-color 0.3s",
    fontFamily: customFont,
    boxShadow: `4px 4px 0px ${darkPurple}`,
  };
  const selectedButtonStyle = {
    // ⭐️ 성별 선택 버튼 스타일
    ...buttonStyle,
    backgroundColor: darkPurple,
    color: white,
    boxShadow: `inset 4px 4px 0px ${deppDarkPurple}`,
  };

  // ------------------------------------
  // 5. 컴포넌트 렌더링 (적용)
  // ------------------------------------
  return (
    <div style={containerStyle}>
      <style>{fontFaceCss}</style>
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
            onChange={createHandleChange(setcustomer_id)} // ⭐️ useCallback 함수 적용
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
            onChange={createHandleChange(setEmail)} // ⭐️ useCallback 함수 적용
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
            onChange={createHandleChange(setPassword)} // ⭐️ useCallback 함수 적용
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
            onChange={createHandleChange(setConfirmPassword)} // ⭐️ useCallback 함수 적용
          />
        </div>

        {/* 5. 생년월일 (Date) 및 나이 (자동 계산) */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_birthDate" style={labelStyle}>
            생년월일
          </label>
          <input
            type="date"
            id="reg_birthDate"
            style={inputStyle}
            value={birthDate}
            onChange={handleBirthDateChange} // ⭐️ 나이 계산 로직 포함
          />
          <div style={{ color: darkPurple }}>
            계산된 나이: <span style={{ color: darkPurple }}>{age} 세</span>
          </div>
        </div>

        {/* 6. 성별 (버튼 선택) */}
        <div style={inputGroupStyle}>
          <label style={labelStyle}>성별</label>
          <div
            style={{
              display: "flex",
              justifyContent: "space-around",
              gap: "10px",
            }}
          >
            <button
              type="button"
              style={gender === 'M' ? selectedButtonStyle : buttonStyle} 
              onClick={() => handleGenderSelect('M')}
            >
              남자
            </button>
            <button
              type="button"
              style={gender === null? selectedButtonStyle : buttonStyle}
              onClick={() => handleGenderSelect(null)}
            >
              비공개
            </button>
            <button
              type="button"
              style={gender === 'F' ? selectedButtonStyle : buttonStyle}
              onClick={() => handleGenderSelect('F')}
            >
              여자
            </button>
          </div>
        </div>
        {/* 버튼 영역 */}
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