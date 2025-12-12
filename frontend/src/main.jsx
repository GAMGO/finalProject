import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import App from "./App.jsx"; // 메인 앱 레이아웃
import AuthPage from "./pages/AuthPage.jsx"; // 로그인/회원가입 페이지
import AuthCheck from "./pages/AuthCheck.jsx"; // 인증 상태 확인 (AuthCheck)
import Recovery from './pages/RecoveringPage.jsx' //계정복구
import { ThemeProvider } from "./theme/ThemeContext"; // ✅ 추가

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <ThemeProvider> 
        <Routes>
          <Route 
              path="/*" 
              element={
                  <AuthCheck>
                      <App />
                  </AuthCheck>
              }
          />    
        </Routes>
      </ThemeProvider>
    </BrowserRouter>
  </StrictMode>
);
