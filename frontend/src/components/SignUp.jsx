// Signup.jsx
import { useState } from "react";
import axios from "axios";

function Signup() {
  const [form, setForm] = useState({
    customersId: "",
    password: "",
    name: "",
    birth: "",
    gender: 0,
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
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
    <form onSubmit={handleSubmit}>
      <input name="customersId" placeholder="아이디" onChange={handleChange} />
      <input
        type="password"
        name="password"
        placeholder="비밀번호"
        onChange={handleChange}
      />
      <input name="name" placeholder="이름" onChange={handleChange} />
      <input
        name="birth"
        placeholder="생년월일 (YYYY-MM-DD)"
        onChange={handleChange}
      />

      <select name="gender" onChange={handleChange}>
        <option value={0}>남성</option>
        <option value={1}>여성</option>
      </select>

      <button type="submit">회원가입</button>
    </form>
  );
}

export default Signup;
