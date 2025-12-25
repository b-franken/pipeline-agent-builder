"use client";

import type { TaskResponse } from "@/lib/api";
import styles from "./ResultPanel.module.css";

interface ResultPanelProps {
  result: TaskResponse | null;
  error: string | null;
}

export default function ResultPanel({ result, error }: ResultPanelProps) {
  if (error) {
    return (
      <div className={`${styles.panel} ${styles.error}`}>
        <div className={styles.header}>Error</div>
        <div className={styles.content}>{error}</div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className={styles.panel}>
        <div className={styles.header}>Result</div>
        <div className={styles.empty}>Send a task to see the result here.</div>
      </div>
    );
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span>Result</span>
        <span className={styles.meta}>
          Task: {result.task_id} | Agent: {result.agent}
        </span>
      </div>
      <div className={styles.content}>{result.result}</div>
    </div>
  );
}
