"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import LogPanel from "@/components/LogPanel";
import KnowledgePanel from "@/components/KnowledgePanel";
import SettingsPanel from "@/components/SettingsPanel";
import AgentEditor from "@/components/AgentEditor";
import ExecutionGraph from "@/components/ExecutionGraph";
import PipelineEditor from "@/components/PipelineEditor";
import TeamEditor from "@/components/TeamEditor";
import MCPServerManager from "@/components/MCPServerManager";
import DraggableDashboard, { resetDashboardLayout } from "@/components/DraggableDashboard";
import ChatPanel from "@/components/ChatPanel";
import { fetchAgents, runTask, runPipeline, createWebSocket, fetchPipelines, type Agent, type LogEvent, type TaskResponse, type Pipeline } from "@/lib/api";
import styles from "./page.module.css";

type Tab = "dashboard" | "creator" | "settings";
type CreatorTab = "pipelines" | "agents" | "teams" | "mcp";

export default function Dashboard() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [result, setResult] = useState<TaskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>("dashboard");
  const [creatorTab, setCreatorTab] = useState<CreatorTab>("pipelines");
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(null);
  const [isEditingLayout, setIsEditingLayout] = useState(false);

  useEffect(() => {
    fetchAgents()
      .then(setAgents)
      .catch((err) => console.error("Failed to fetch agents:", err));
    fetchPipelines()
      .then(setPipelines)
      .catch((err) => console.error("Failed to fetch pipelines:", err));
  }, []);

  useEffect(() => {
    const clientId = `dashboard-${Date.now()}`;
    const ws = createWebSocket(
      clientId,
      (event) => {
        setLogs((prev) => [...prev, event]);
        if (event.type === "agent_start") {
          setActiveAgent(event.agent);
        } else if (event.type === "agent_end" || event.type === "task_end") {
          setActiveAgent(null);
        }
      },
      () => setWsConnected(false)
    );

    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);

    return () => ws.close();
  }, []);

  const handleSubmit = useCallback(async (task: string) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setLogs([]);

    try {
      let response: TaskResponse;
      if (selectedPipelineId) {
        response = await runPipeline(selectedPipelineId, task);
      } else {
        response = await runTask(task);
      }
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
      setActiveAgent(null);
    }
  }, [selectedPipelineId]);

  const dashboardWidgets = useMemo(() => [
    {
      id: "chat",
      title: "Chat",
      component: (
        <ChatPanel
          onSubmit={handleSubmit}
          isLoading={isLoading}
          result={result}
          error={error}
          logs={logs}
          selectedPipeline={selectedPipelineId}
          pipelines={pipelines}
          onPipelineChange={setSelectedPipelineId}
        />
      ),
    },
    {
      id: "execution",
      title: isLoading ? "Execution Flow (Running...)" : "Execution Flow",
      component: (
        <ExecutionGraph
          agents={
            selectedPipelineId
              ? (() => {
                  const pipeline = pipelines.find((p) => p.id === selectedPipelineId);
                  if (!pipeline) return agents.map((a) => ({ name: a.name, description: a.description, status: "idle" as const }));
                  return (pipeline.nodes || [])
                    .filter((n) => (n as { type?: string }).type === "agent")
                    .map((n) => {
                      const data = n.data as { label?: string; description?: string; agentName?: string };
                      const agentName = data.agentName || data.label || "";
                      return {
                        name: data.label || agentName,
                        description: data.description || "",
                        status: activeAgent === agentName ? "active" as const : "idle" as const,
                      };
                    });
                })()
              : agents.map((a) => ({
                  name: a.name,
                  description: a.description,
                  status: activeAgent === a.name ? "active" : a.status === "working" ? "active" : "idle",
                }))
          }
          events={logs.map((l) => ({
            type: l.type as "agent_start" | "agent_end" | "handoff" | "tool_call" | "task_start" | "task_end" | "error" | "finish",
            agent: l.agent,
            from: l.from,
            to: l.to,
            tool: l.tool,
            message: l.message,
            timestamp: l.timestamp ? String(l.timestamp) : undefined,
          }))}
          isExecuting={isLoading}
          pipelineNodes={selectedPipelineId ? pipelines.find((p) => p.id === selectedPipelineId)?.nodes : undefined}
          pipelineEdges={selectedPipelineId ? pipelines.find((p) => p.id === selectedPipelineId)?.edges : undefined}
        />
      ),
    },
    {
      id: "activity",
      title: `Activity${logs.length > 0 ? ` (${logs.length})` : ""}`,
      component: <LogPanel logs={logs} />,
    },
    {
      id: "knowledge",
      title: "Knowledge Base",
      component: <KnowledgePanel />,
    },
  ], [agents, logs, result, error, isLoading, activeAgent, selectedPipelineId, pipelines, handleSubmit]);

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <div className={styles.logoIcon}>K</div>
          <h1>De Kantoorkiller</h1>
        </div>
        <nav className={styles.nav}>
          <button
            className={`${styles.navBtn} ${activeTab === "dashboard" ? styles.active : ""}`}
            onClick={() => setActiveTab("dashboard")}
          >
            Dashboard
          </button>
          <button
            className={`${styles.navBtn} ${activeTab === "creator" ? styles.active : ""}`}
            onClick={() => setActiveTab("creator")}
          >
            Creator
          </button>
          <button
            className={`${styles.navBtn} ${activeTab === "settings" ? styles.active : ""}`}
            onClick={() => setActiveTab("settings")}
          >
            Settings
          </button>
        </nav>
        <div className={styles.headerRight}>
          {activeTab === "dashboard" && (
            <div className={styles.layoutControls}>
              <button
                className={`${styles.layoutBtn} ${isEditingLayout ? styles.layoutBtnActive : ""}`}
                onClick={() => setIsEditingLayout(!isEditingLayout)}
                title={isEditingLayout ? "Klaar" : "Layout aanpassen"}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="7" height="7" rx="1" />
                  <rect x="14" y="3" width="7" height="7" rx="1" />
                  <rect x="3" y="14" width="7" height="7" rx="1" />
                  <rect x="14" y="14" width="7" height="7" rx="1" />
                </svg>
              </button>
              {isEditingLayout && (
                <button
                  className={styles.resetBtn}
                  onClick={resetDashboardLayout}
                  title="Reset layout"
                >
                  Reset
                </button>
              )}
            </div>
          )}
          <div className={styles.status}>
            <span className={`${styles.dot} ${wsConnected ? styles.connected : ""}`} />
            {wsConnected ? "Live" : "Offline"}
          </div>
        </div>
      </header>

      {activeTab === "dashboard" && (
        <DraggableDashboard widgets={dashboardWidgets} isEditing={isEditingLayout} />
      )}

      {activeTab === "creator" && (
        <div className={styles.creatorLayout}>
          <div className={styles.creatorNav}>
            <button
              className={`${styles.creatorTab} ${creatorTab === "pipelines" ? styles.activeCreatorTab : ""}`}
              onClick={() => setCreatorTab("pipelines")}
            >
              Pipelines
            </button>
            <button
              className={`${styles.creatorTab} ${creatorTab === "agents" ? styles.activeCreatorTab : ""}`}
              onClick={() => setCreatorTab("agents")}
            >
              Agents
            </button>
            <button
              className={`${styles.creatorTab} ${creatorTab === "teams" ? styles.activeCreatorTab : ""}`}
              onClick={() => setCreatorTab("teams")}
            >
              Teams
            </button>
            <button
              className={`${styles.creatorTab} ${creatorTab === "mcp" ? styles.activeCreatorTab : ""}`}
              onClick={() => setCreatorTab("mcp")}
            >
              MCP Servers
            </button>
          </div>

          <div className={styles.creatorContent}>
            {creatorTab === "pipelines" && (
              <div className={styles.creatorPanel}>
                <div className={styles.creatorHeader}>
                  <h2>Pipeline Builder</h2>
                  <p>Design custom agent workflows</p>
                </div>
                <PipelineEditor
                  agents={agents.map((a) => ({
                    name: a.name,
                    description: a.description,
                  }))}
                  onPipelineSelect={(pipelineId) => {
                    setSelectedPipelineId(pipelineId);
                    fetchPipelines().then(setPipelines);
                  }}
                />
              </div>
            )}

            {creatorTab === "agents" && (
              <div className={styles.creatorPanel}>
                <AgentEditor />
              </div>
            )}

            {creatorTab === "teams" && (
              <div className={styles.creatorPanel}>
                <div className={styles.creatorHeader}>
                  <h2>Team Manager</h2>
                  <p>Create agent teams for pipelines</p>
                </div>
                <TeamEditor />
              </div>
            )}

            {creatorTab === "mcp" && (
              <div className={styles.creatorPanel}>
                <div className={styles.creatorHeader}>
                  <h2>MCP Server Manager</h2>
                  <p>Configure Model Context Protocol servers for agent tools</p>
                </div>
                <MCPServerManager />
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "settings" && (
        <div className={styles.settingsLayout}>
          <SettingsPanel />
        </div>
      )}
    </main>
  );
}
