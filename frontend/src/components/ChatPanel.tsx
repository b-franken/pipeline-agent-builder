"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import styles from "./ChatPanel.module.css";

interface Message {
  id: string;
  type: "user" | "assistant" | "system" | "error";
  content: string;
  timestamp: Date;
  isLoading?: boolean;
}

interface ChatPanelProps {
  onSubmit: (task: string) => void;
  isLoading: boolean;
  result: { result?: string } | null;
  error: string | null;
  logs: { type: string; message?: string; agent?: string }[];
  selectedPipeline: string | null;
  pipelines: { id: string; name: string }[];
  onPipelineChange: (id: string | null) => void;
}

export default function ChatPanel({
  onSubmit,
  isLoading,
  result,
  error,
  logs,
  selectedPipeline,
  pipelines,
  onPipelineChange,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [lastResultId, setLastResultId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Handle new result
  useEffect(() => {
    if (result?.result && lastResultId !== result.result.slice(0, 50)) {
      setMessages((prev) => {
        // Remove loading message
        const filtered = prev.filter((m) => !m.isLoading);
        return [
          ...filtered,
          {
            id: `result-${Date.now()}`,
            type: "assistant",
            content: result.result || "",
            timestamp: new Date(),
          },
        ];
      });
      setLastResultId(result.result.slice(0, 50));
    }
  }, [result, lastResultId]);

  // Handle error
  useEffect(() => {
    if (error) {
      setMessages((prev) => {
        const filtered = prev.filter((m) => !m.isLoading);
        return [
          ...filtered,
          {
            id: `error-${Date.now()}`,
            type: "error",
            content: error,
            timestamp: new Date(),
          },
        ];
      });
    }
  }, [error]);

  // Handle loading state
  useEffect(() => {
    if (isLoading) {
      const hasLoading = messages.some((m) => m.isLoading);
      if (!hasLoading) {
        setMessages((prev) => [
          ...prev,
          {
            id: `loading-${Date.now()}`,
            type: "system",
            content: "Processing...",
            timestamp: new Date(),
            isLoading: true,
          },
        ]);
      }
    }
  }, [isLoading, messages]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        type: "user",
        content: input.trim(),
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);
      onSubmit(input.trim());
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("nl-NL", { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <select
          className={styles.pipelineSelector}
          value={selectedPipeline || ""}
          onChange={(e) => onPipelineChange(e.target.value || null)}
        >
          <option value="">Supervisor Mode</option>
          {pipelines.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.messages}>
        {messages.length === 0 ? (
          <div className={styles.empty}>
            <div className={styles.emptyIcon}>K</div>
            <div className={styles.emptyTitle}>De Kantoorkiller</div>
            <div className={styles.emptyText}>
              Stel een vraag of geef een opdracht aan de agents
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`${styles.message} ${styles[message.type]} ${
                message.isLoading ? styles.loading : ""
              }`}
            >
              {message.type === "user" ? (
                <div className={styles.userBubble}>
                  <div className={styles.messageContent}>{message.content}</div>
                  <div className={styles.messageTime}>{formatTime(message.timestamp)}</div>
                </div>
              ) : message.type === "assistant" ? (
                <div className={styles.assistantBubble}>
                  <div className={styles.assistantAvatar}>K</div>
                  <div className={styles.assistantContent}>
                    <div className={styles.messageContent}>{message.content}</div>
                    <div className={styles.messageTime}>{formatTime(message.timestamp)}</div>
                  </div>
                </div>
              ) : message.type === "error" ? (
                <div className={styles.errorBubble}>
                  <div className={styles.errorIcon}>!</div>
                  <div className={styles.messageContent}>{message.content}</div>
                </div>
              ) : (
                <div className={styles.systemBubble}>
                  {message.isLoading && <span className={styles.loadingDots} />}
                  <span>{message.content}</span>
                </div>
              )}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className={styles.inputArea} onSubmit={handleSubmit}>
        <textarea
          ref={inputRef}
          className={styles.input}
          placeholder="Typ je bericht..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          rows={1}
        />
        <button
          type="submit"
          className={styles.sendBtn}
          disabled={isLoading || !input.trim()}
        >
          <span className={styles.sendIcon}>↑</span>
        </button>
      </form>
    </div>
  );
}
