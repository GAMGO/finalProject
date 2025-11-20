import KakaoMap from "./components/KakaoMap";
import "./App.css";

export default function App() {
  return (
    <div className="app">
      <aside className="side-bar">
        <div className="side-logo">LOGO</div>
      </aside>

      <main className="main-content">
        <KakaoMap />
      </main>
    </div>
  );
}
