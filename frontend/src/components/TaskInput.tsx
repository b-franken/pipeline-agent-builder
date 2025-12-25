"use client";

import { useState, FormEvent } from "react";
import styles from "./TaskInput.module.css";

interface TaskInputProps {
  onSubmit: (task: string) => void;
  isLoading: boolean;
}

export default function TaskInput({ onSubmit, isLoading }: TaskInputProps) {
  const [task, setTask] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (task.trim() && !isLoading) {
      onSubmit(task.trim());
      setTask("");
    }
  };

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <input
        type="text"
        className={styles.input}
        placeholder="Enter a task for the agents..."
        value={task}
        onChange={(e) => setTask(e.target.value)}
        disabled={isLoading}
      />
      <button type="submit" className={styles.button} disabled={isLoading || !task.trim()}>
        {isLoading ? "Running..." : "Send"}
      </button>
    </form>
  );
}
