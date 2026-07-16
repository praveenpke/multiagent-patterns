import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchPattern, resumePattern, runPattern } from "../api";
import Mermaid from "../components/Mermaid";
import Timeline from "../components/Timeline";
import type { Ev, Pattern } from "../types";

interface PendingApproval {
  threadId: string;
  proposal: string;
  question: string;
}

export default function PatternDetail() {
  const { name = "" } = useParams();
  const [pattern, setPattern] = useState<Pattern | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [events, setEvents] = useState<Ev[]>([]);
  const [running, setRunning] = useState(false);
  const [approval, setApproval] = useState<PendingApproval | null>(null);
  const [feedback, setFeedback] = useState("");
  const [finalState, setFinalState] = useState<Record<string, any> | null>(null);

  useEffect(() => {
    fetchPattern(name)
      .then((p) => {
        setPattern(p);
        setInput(p.default_text);
      })
      .catch((e) => setError(String(e)));
  }, [name]);

  const onEvent = (event: Ev) => {
    setEvents((prev) => [...prev, event]);
    if (event.type === "interrupt") {
      const payload = event.data.payload ?? {};
      setApproval({
        threadId: event.data.thread_id,
        proposal: String(payload.proposal ?? JSON.stringify(payload)),
        question: String(payload.question ?? "Approve?"),
      });
    }
    if (event.type === "run_end" && !event.data.interrupted) {
      setFinalState(event.data.final_state ?? null);
    }
  };

  const run = async () => {
    setEvents([]);
    setFinalState(null);
    setApproval(null);
    setRunning(true);
    try {
      await runPattern(name, input, onEvent);
    } catch (e) {
      onEvent({ type: "error", node: null, agent: null, data: { message: String(e) }, ts: 0 });
    } finally {
      setRunning(false);
    }
  };

  const decide = async (approved: boolean) => {
    if (!approval) return;
    const threadId = approval.threadId;
    setApproval(null);
    setRunning(true);
    try {
      await resumePattern(name, threadId, approved, feedback, onEvent);
    } catch (e) {
      onEvent({ type: "error", node: null, agent: null, data: { message: String(e) }, ts: 0 });
    } finally {
      setRunning(false);
      setFeedback("");
    }
  };

  if (error) return <div className="error-box">{error}</div>;
  if (!pattern) return <div className="loading">loading pattern…</div>;

  return (
    <>
      <Link to="/" className="crumb">← all patterns</Link>
      <div className="detail-head">
        <h1>{pattern.title}</h1>
        <span className={`badge ${pattern.category}`}>{pattern.category}</span>
        {pattern.supports_hitl && <span className="badge hitl">HITL</span>}
      </div>
      <p className="detail-desc">{pattern.description}</p>

      <div className="detail-grid">
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div className="panel">
            <header>graph</header>
            <div className="panel-body">
              <Mermaid chart={pattern.mermaid} />
            </div>
          </div>

          <div className="panel">
            <header>input</header>
            <div className="panel-body run-controls">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={running}
                spellCheck={false}
              />
              <div className="run-row">
                <button className="primary" onClick={run} disabled={running || !!approval}>
                  {running ? "running…" : "Run"}
                </button>
                {running && (
                  <span className="status">
                    <span className="running-dot" />
                    streaming events
                  </span>
                )}
                {approval && <span className="status">paused — decision required</span>}
              </div>

              {approval && (
                <div className="approval">
                  <h4>human gate · {approval.question}</h4>
                  <div className="proposal">{approval.proposal}</div>
                  <input
                    placeholder="optional feedback for the agent"
                    value={feedback}
                    onChange={(e) => setFeedback(e.target.value)}
                  />
                  <div className="buttons">
                    <button className="approve" onClick={() => decide(true)}>
                      Approve
                    </button>
                    <button className="reject" onClick={() => decide(false)}>
                      Reject
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {finalState && (
            <div className="panel final-state">
              <header>final state</header>
              <pre>{JSON.stringify(finalState, null, 2)}</pre>
            </div>
          )}
        </div>

        <div className="panel">
          <header>execution timeline</header>
          <Timeline events={events} />
        </div>
      </div>
    </>
  );
}
