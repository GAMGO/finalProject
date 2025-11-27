import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
// import './index.css'
import App from "./App.jsx";
import AuthPage from "./pages/AuthPage.jsx";
import Check from "./pages/AuthCheck.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/auth" element={<AuthPage />} />
        <Route path="/app" element={<App />} />
        <Route path="/" element={<Check />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>
);
