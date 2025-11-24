import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import FavoritePage from "./pages/FavoritePage";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <FavoritePage />
  </React.StrictMode>
);
