import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import UserProfilePage from "./pages/UserProfilePage";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <UserProfilePage />
  </React.StrictMode>
);
