import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
// import './index.css'
import App from "./App.jsx"; // 메인 앱 레이아웃
import AuthPage from "./pages/AuthPage.jsx"; // 로그인/회원가입 페이지
import Check from "./pages/AuthCheck.jsx"; // 인증 상태 확인 (AuthCheck)

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        {/* 1. 인증 경로: 토큰이 없을 때 접근 (보호되지 않음) */}
        <Route path="/auth" element={<AuthPage />} />
        
        {/* 2. 메인 애플리케이션 경로: 루트 '/'와 그 하위 모든 경로를 처리합니다. */}
        {/* Check(AuthCheck) 컴포넌트가 내부의 <App /> 렌더링을 허용하거나 /auth로 리디렉션합니다. */}
        <Route 
            path="/*" 
            element={
                <Check>
                    <App />
                </Check>
            } 
        />

        {/* 3. 예외적인 경우 (예: /old-page 등으로 접근 시) '/'로 리디렉션 */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>
);