export interface Pattern {
  name: string;
  title: string;
  category: "orchestration" | "control-flow" | "memory";
  description: string;
  mermaid: string;
  default_text: string;
  supports_hitl: boolean;
}

export type EventType =
  | "run_start"
  | "node_start"
  | "node_end"
  | "agent_message"
  | "tool_call"
  | "tool_result"
  | "interrupt"
  | "run_end"
  | "error";

export interface Ev {
  type: EventType;
  node: string | null;
  agent: string | null;
  data: Record<string, any>;
  ts: number;
}
