import mermaid from "mermaid";
import { useEffect, useRef } from "react";

mermaid.initialize({
  startOnLoad: false,
  theme: "dark",
  themeVariables: {
    darkMode: true,
    background: "#141927",
    primaryColor: "#1b2233",
    primaryBorderColor: "#32405e",
    primaryTextColor: "#dde3f0",
    lineColor: "#5a6479",
    fontFamily: "JetBrains Mono, Consolas, monospace",
    fontSize: "13px",
  },
});

let renderCounter = 0;

export default function Mermaid({ chart }: { chart: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    const id = `mmd-${renderCounter++}`;
    mermaid
      .render(id, chart)
      .then(({ svg }) => {
        if (!cancelled && ref.current) ref.current.innerHTML = svg;
      })
      .catch((err) => {
        if (!cancelled && ref.current)
          ref.current.textContent = `diagram error: ${err}`;
      });
    return () => {
      cancelled = true;
    };
  }, [chart]);

  return <div className="diagram" ref={ref} />;
}
