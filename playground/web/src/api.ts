import type { Ev, Pattern } from "./types";

export async function fetchPatterns(): Promise<Pattern[]> {
  const res = await fetch("/api/patterns");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchPattern(name: string): Promise<Pattern> {
  const res = await fetch(`/api/patterns/${name}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/** POST to an SSE endpoint and invoke onEvent for every streamed event. */
async function streamSSE(
  url: string,
  body: unknown,
  onEvent: (event: Ev) => void,
): Promise<void> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice("data: ".length)));
    }
  }
}

export function runPattern(
  name: string,
  text: string,
  onEvent: (event: Ev) => void,
): Promise<void> {
  return streamSSE(`/api/patterns/${name}/run`, { text }, onEvent);
}

export function resumePattern(
  name: string,
  threadId: string,
  approved: boolean,
  feedback: string,
  onEvent: (event: Ev) => void,
): Promise<void> {
  return streamSSE(
    `/api/patterns/${name}/resume`,
    { thread_id: threadId, approved, feedback },
    onEvent,
  );
}
