import { useEffect, useRef } from "react";
import type { Ev } from "../types";

const AGENT_PALETTE = [
  "#5eead4",
  "#a78bfa",
  "#f472b6",
  "#60a5fa",
  "#fbbf24",
  "#4ade80",
  "#fb923c",
  "#22d3ee",
  "#e879f9",
  "#a3e635",
];

function agentColor(agent: string): string {
  let hash = 0;
  for (const ch of agent) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return AGENT_PALETTE[hash % AGENT_PALETTE.length];
}

function eventBody(event: Ev): string {
  switch (event.type) {
    case "run_start":
      return "run started";
    case "run_end":
      return event.data.interrupted
        ? "run paused - waiting for human decision"
        : "run finished";
    case "node_start":
      return event.node ?? "";
    case "node_end":
      return event.node ?? "";
    case "agent_message":
      return String(event.data.content ?? "");
    case "tool_call":
      return `${event.data.tool}(${JSON.stringify(event.data.args)})`;
    case "tool_result":
      return `${event.data.tool} -> ${String(event.data.result)}`;
    case "interrupt":
      return `paused: ${JSON.stringify(event.data.payload)}`;
    case "error":
      return String(event.data.message ?? "error");
  }
}

export default function Timeline({ events }: { events: Ev[] }) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [events.length]);

  return (
    <div className="timeline">
      {events.map((event, i) => (
        <div key={i} className={`evt ${event.type}`}>
          <span className="tag">{event.type.replace("_", " ")}</span>
          <span className="body">
            {event.agent && (
              <span className="agent-chip" style={{ color: agentColor(event.agent) }}>
                {event.agent}
              </span>
            )}
            {eventBody(event)}
          </span>
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
