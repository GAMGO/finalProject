import { useEffect, useState } from "react";
import axios from "axios";

function Signup() {
  const [form, setForm] = useState({
    customersId: "",
    password: "",
    name: "",
    birth: "",
    gender: 0,
  });

  useEffect(() => {
    /* 구글 초기화 */
    window.google.accounts.id.initialize({
      client_id: "YOUR_GOOGLE_CLIENT_ID",
      callback: handleGoogleLogin, // 구글 토큰 수신
    });

    window.google.accounts.id.renderButton(
      document.getElementById("googleLoginBtn"),
      { theme: "outline", size: "large" }
    );
  }, []);

  // 구글 로그인 callback
  const handleGoogleLogin = async (response) => {
    try {
      const res = await axios.post("/api/auth/google", {
        credential: response.credential, // Google ID Token
      });

      alert("구글 회원가입/로그인 성공!");

      // JWT 저장
      localStorage.setItem("accessToken", res.data.accessToken);
    } catch (err) {
      console.error(err);
      alert("구글 로그인 오류");
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;

    setForm({
      ...form,
      [name]: name === "gender" ? Number(value) : value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      await axios.post("/api/auth/signup", form);
      alert("회원가입 성공!");
    } catch (err) {
      alert("오류: " + err.response.data.message);
    }
  };

  return (
    <>
      <form onSubmit={handleSubmit}>
        <input
          name="customersId"
          placeholder="아이디"
          onChange={handleChange}
        />
        <input
          type="password"
          name="password"
          placeholder="비밀번호"
          onChange={handleChange}
        />
        <input name="name" placeholder="이름" onChange={handleChange} />
        <input name="birth" placeholder="YYYY-MM-DD" onChange={handleChange} />

        <select name="gender" onChange={handleChange}>
          <option value={0}>남성</option>
          <option value={1}>여성</option>
        </select>

        <button type="submit">회원가입</button>
      </form>

      <div style={{ marginTop: "20px" }}>
        <div id="googleLoginBtn"></div>
      </div>
    </>
  );
}

export default Signup;
