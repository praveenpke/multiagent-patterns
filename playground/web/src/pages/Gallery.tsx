import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchPatterns } from "../api";
import type { Pattern } from "../types";

const CATEGORIES: { key: Pattern["category"]; label: string }[] = [
  { key: "orchestration", label: "Orchestration" },
  { key: "control-flow", label: "Control Flow" },
  { key: "memory", label: "Memory" },
];

export default function Gallery() {
  const [patterns, setPatterns] = useState<Pattern[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPatterns().then(setPatterns).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error-box">Failed to load patterns: {error}</div>;
  if (!patterns) return <div className="loading">loading patterns…</div>;

  return (
    <>
      <div className="hero">
        <h1>Multi-agent patterns, live.</h1>
        <p>
          Ten production-grade LangGraph orchestration patterns with typed state,
          guardrails, and a structured event stream. Pick one, run it, and watch
          the graph execute node by node — no API key required.
        </p>
      </div>
      {CATEGORIES.map(({ key, label }) => {
        const group = patterns.filter((p) => p.category === key);
        if (!group.length) return null;
        return (
          <section className="category" key={key}>
            <h2>{label}</h2>
            <div className="cards">
              {group.map((p) => (
                <Link to={`/p/${p.name}`} key={p.name} className="card">
                  <div className="card-head">
                    <h3>{p.title}</h3>
                    <span className={`badge ${p.category}`}>{p.category}</span>
                    {p.supports_hitl && <span className="badge hitl">HITL</span>}
                  </div>
                  <p>{p.description}</p>
                  <span className="go">run pattern →</span>
                </Link>
              ))}
            </div>
          </section>
        );
      })}
    </>
  );
}
