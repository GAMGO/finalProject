import React, { useState, useRef, useEffect } from "react";
// assets 폴더가 src 안에 있다고 가정: src/assets/ham_light.png
import hamLight from "../assets/ham_light.png";

function Header() {
  const [open, setOpen] = useState(false);
  const menuRef = useRef(null);

  // 바깥 클릭 시 드롭다운 닫기
  useEffect(() => {
    function handleClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    if (open) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const handleSelect = (type) => {
    console.log("selected:", type);
    setOpen(false);
    // TODO: 여기서 라우팅 / 화면 전환 붙이면 됨.
  };

  return (
    <header className="header">
      <div className="header-inner">
        {/* 왼쪽: 햄버거 + 드롭다운 */}
        <div className="header-left" ref={menuRef}>
          <button
            className="menu-button"
            type="button"
            onClick={() => setOpen((v) => !v)}
          >
            <img src={hamLight} alt="메뉴" className="menu-icon" />
          </button>

          {open && (
            <div className="menu-dropdown">
              <button
                type="button"
                className="menu-item"
                onClick={() => handleSelect("community")}
              >
                커뮤니티
              </button>
              <button
                type="button"
                className="menu-item"
                onClick={() => handleSelect("settings")}
              >
                설정
              </button>
            </div>
          )}
        </div>

        {/* 중앙: 로고 자리 */}
        <div className="header-center">
          <div className="logo-placeholder">LOGO</div>
        </div>

        {/* 오른쪽: 나중에 프로필/로그인 등 넣을 자리 */}
        <div className="header-right" />
      </div>
    </header>
  );
}

export default Header;
