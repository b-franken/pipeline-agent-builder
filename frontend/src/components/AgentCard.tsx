"use client";

import type { Agent } from "@/lib/api";
import styles from "./AgentCard.module.css";

interface AgentCardProps {
  agent: Agent;
  isActive?: boolean;
}

export default function AgentCard({ agent, isActive }: AgentCardProps) {
  const statusClass = isActive ? styles.active : styles[agent.status];

  return (
    <div className={`${styles.card} ${statusClass}`}>
      <div className={styles.header}>
        <span className={styles.name}>{agent.name}</span>
        <span className={`${styles.status} ${statusClass}`}>
          {isActive ? "working" : agent.status}
        </span>
      </div>
      <p className={styles.description}>{agent.description}</p>
    </div>
  );
}
