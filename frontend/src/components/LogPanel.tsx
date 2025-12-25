"use client";

import { useEffect, useRef } from "react";
import type { LogEvent } from "@/lib/api";
import styles from "./LogPanel.module.css";

interface LogPanelProps {
  logs: LogEvent[];
}

function formatTime(timestamp: number | undefined): string {
  if (!timestamp) return "--:--:--";
  // Handle both seconds (from Python time.time()) and milliseconds
  const ms = timestamp > 1e12 ? timestamp : timestamp * 1000;
  return new Date(ms).toLocaleTimeString();
}

function getEventIcon(type: string): string {
  switch (type) {
    case "task_start":
      return ">";
    case "task_end":
      return "<";
    case "agent_start":
      return "+";
    case "agent_response":
      return "<-";
    case "agent_end":
      return "-";
    case "handoff":
      return "=>";
    case "tool_call":
      return "*";
    case "tool_result":
      return "<*";
    case "finish":
      return "[x]";
    case "error":
      return "!";
    default:
      return " ";
  }
}

function getEventClass(type: string): string {
  if (type === "error") return styles.error;
  if (type === "handoff") return styles.handoff;
  if (type === "finish") return styles.finish;
  if (type.includes("start")) return styles.start;
  if (type.includes("end") || type.includes("result")) return styles.end;
  if (type.includes("tool")) return styles.tool;
  if (type === "agent_response") return styles.response;
  return "";
}

export default function LogPanel({ logs }: LogPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span>Event Log</span>
        <span className={styles.count}>{logs.length}</span>
      </div>
      <div className={styles.logs}>
        {logs.length === 0 ? (
          <div className={styles.empty}>No events yet. Send a task to see activity.</div>
        ) : (
          logs.map((log, i) => (
            <div key={i} className={`${styles.entry} ${getEventClass(log.type)}`}>
              <span className={styles.time}>{formatTime(log.timestamp)}</span>
              <span className={styles.icon}>{getEventIcon(log.type)}</span>
              <span className={styles.agent}>[{log.agent}]</span>
              <span className={styles.message}>{log.message}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
