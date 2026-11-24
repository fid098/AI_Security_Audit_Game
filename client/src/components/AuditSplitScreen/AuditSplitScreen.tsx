import { useEffect, useState } from "react";
import "./AuditSplitScreen.css";

// Mounted only while an audit runs; unmounting discards the animation state.
interface AuditSplitScreenProps {
  findingsCount: number;
}

const LOG_LINES = [
  "Initializing Hacktron scan engine...",
  "Enumerating ship subsystems...",
  "Loading signature database v2.1.0...",
  "Analyzing input sinks...",
  "Tracing data flows...",
  "Checking outbound request policies...",
  "Scanning for unsafe deserialization...",
  "Verifying query parameterization...",
  "Mapping vulnerable execution paths...",
  "Compiling security report...",
  "Requesting Claude summary...",
];

export function AuditSplitScreen({ findingsCount }: AuditSplitScreenProps) {
  const [lines, setLines] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    // Random is impure, so it belongs in the effect rather than in render.
    const seed = Math.floor(Math.random() * 1000);
    let index = 0;
    const logTimer = setInterval(() => {
      const nextLine = LOG_LINES[index % LOG_LINES.length];
      setLines((prev) => [...prev.slice(-10), `[${seed + index}] ${nextLine}`]);
      index += 1;
    }, 280);

    const progressTimer = setInterval(() => {
      setProgress((prev) => (prev >= 100 ? 100 : prev + 4));
    }, 260);

    return () => {
      clearInterval(logTimer);
      clearInterval(progressTimer);
    };
  }, []);

  return (
    <div className="audit-split">
      <section className="audit-left">
        <div className="audit-terminal-header">LIVE SECURITY SCAN</div>
        <div className="audit-terminal-body">
          {lines.map((line, idx) => (
            <div key={`${line}-${idx}`} className="audit-terminal-line">
              {line}
            </div>
          ))}
        </div>
      </section>

      <section className="audit-right">
        <div className="audit-metric">
          <div className="audit-progress-ring">
            <svg viewBox="0 0 120 120">
              <circle className="ring-bg" cx="60" cy="60" r="50" />
              <circle
                className="ring-progress"
                cx="60"
                cy="60"
                r="50"
                style={{
                  strokeDasharray: 314,
                  strokeDashoffset: 314 - (314 * progress) / 100,
                }}
              />
            </svg>
            <div className="ring-value">{progress}%</div>
          </div>
          <div className="audit-label">SCAN PROGRESS</div>
        </div>

        <div className="audit-metric">
          <div className="audit-count">{findingsCount}</div>
          <div className="audit-label">FINDINGS IDENTIFIED</div>
        </div>

        <div className="audit-status">
          <span className="audit-dot" />
          <span>ANALYSIS IN PROGRESS</span>
        </div>
      </section>
    </div>
  );
}
