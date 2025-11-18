// src/components/SignUpButton.jsx
import { useNavigate } from "react-router-dom";
import SignUp from "../assets/signup.png";

export default function SignUpButton({ open }) {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate("/signup");
  };

  // Sidebar가 펼쳐진 경우 숨김
  if (open) return null;

  return (
    <button type="button" className="collapsed-new-btn" onClick={handleClick}>
      <img src={SignUp} alt="회원가입" className="collapsed-SignUp" />
    </button>
  );
}
