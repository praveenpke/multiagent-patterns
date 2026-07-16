import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import Gallery from "./pages/Gallery";
import PatternDetail from "./pages/PatternDetail";

export default function App() {
  return (
    <BrowserRouter>
      <div className="shell">
        <header className="topbar">
          <Link to="/" className="logo">
            <span className="bracket">[</span>langgraph-patterns
            <span className="bracket">]</span>
          </Link>
          <span className="sub">multi-agent orchestration playground</span>
          <span className="spacer" />
          <a
            className="gh"
            href="https://github.com/praveenpke/multiagent-patterns"
            target="_blank"
            rel="noreferrer"
          >
            github ↗
          </a>
        </header>
        <Routes>
          <Route path="/" element={<Gallery />} />
          <Route path="/p/:name" element={<PatternDetail />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
