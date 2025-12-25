"use client";

import { useState, useCallback, useRef, useMemo, useEffect, DragEvent } from "react";
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Position,
  Handle,
  useReactFlow,
  ReactFlowProvider,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import styles from "./PipelineEditor.module.css";
import {
  fetchPipelines,
  createPipeline,
  updatePipeline,
  deletePipeline,
  fetchTeams,
  fetchAgentConfigs,
  fetchMCPServers,
  type Pipeline,
  type Team,
  type EdgeType,
  type EdgeCondition,
  type EdgeConfig,
  type ConditionType,
  type ConditionField,
  type AgentConfig,
  type MCPServer,
} from "@/lib/api";

interface AgentInfo {
  name: string;
  description: string;
}

interface PipelineEditorProps {
  agents: AgentInfo[];
  initialNodes?: Node[];
  initialEdges?: Edge[];
  onSave?: (nodes: Node[], edges: Edge[]) => void;
  onPipelineSelect?: (pipelineId: string | null) => void;
}

interface AgentNodeData {
  label: string;
  description?: string;
  agentId: string;
}

interface TeamNodeData {
  teamId: string;
  teamName: string;
  agentIds: string[];
  leadAgentId?: string;
  agents: Array<{ id: string; name: string; description?: string }>;
}

interface ToolNodeData {
  label: string;
  mcpServerId: string;
  serverName: string;
  transport: string;
  description?: string;
}

function TeamNode({ data }: { data: TeamNodeData }) {
  return (
    <div className={styles.orgChart}>
      <div className={styles.orgHeader}>
        <Handle type="target" position={Position.Top} id="top" className={`${styles.handle} ${styles.handleTop}`} />
        <Handle type="target" position={Position.Left} id="left" className={`${styles.handle} ${styles.handleLeft}`} />
        <Handle type="source" position={Position.Right} id="right" className={`${styles.handle} ${styles.handleRight}`} />
        <span className={styles.orgTitle}>{data.teamName}</span>
      </div>
      <div className={styles.orgLine} />
      <div className={styles.orgBranches}>
        {data.agents.map((agent, index) => (
          <div key={agent.id} className={styles.orgBranch}>
            <div className={styles.orgBranchLine} />
            <div className={`${styles.orgAgent} ${agent.id === data.leadAgentId ? styles.orgAgentLead : ""}`}>
              <Handle
                type="target"
                position={Position.Left}
                id={`agent-${agent.id}-in`}
                className={`${styles.handle} ${styles.agentHandleLeft}`}
              />
              <span className={styles.orgAgentName}>{agent.name}</span>
              {agent.id === data.leadAgentId && <span className={styles.orgLeadBadge}>L</span>}
              <Handle
                type="source"
                position={Position.Right}
                id={`agent-${agent.id}-out`}
                className={`${styles.handle} ${styles.agentHandleRight}`}
              />
              <Handle
                type="source"
                position={Position.Bottom}
                id={`agent-${agent.id}-bottom`}
                className={`${styles.handle} ${styles.handleBottom}`}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AgentNode({ data }: { data: AgentNodeData }) {
  return (
    <div className={styles.agentNode}>
      <Handle type="target" position={Position.Top} id="top" className={`${styles.handle} ${styles.handleTop}`} />
      <Handle type="target" position={Position.Left} id="left" className={`${styles.handle} ${styles.handleLeft}`} />
      <Handle type="source" position={Position.Right} id="right" className={`${styles.handle} ${styles.handleRight}`} />
      <Handle type="source" position={Position.Bottom} id="bottom" className={`${styles.handle} ${styles.handleBottom}`} />
      <div className={styles.nodeContent}>
        <div className={styles.nodeName}>{data.label}</div>
        {data.description && (
          <div className={styles.nodeDescription}>{data.description}</div>
        )}
      </div>
    </div>
  );
}

function StartNode() {
  return (
    <div className={styles.startNode}>
      <span>Start</span>
      <Handle type="source" position={Position.Bottom} id="bottom" className={`${styles.handle} ${styles.handleBottom}`} />
      <Handle type="source" position={Position.Right} id="right" className={`${styles.handle} ${styles.handleRight}`} />
      <Handle type="source" position={Position.Left} id="left" className={`${styles.handle} ${styles.handleLeft}`} />
    </div>
  );
}

function EndNode() {
  return (
    <div className={styles.endNode}>
      <Handle type="target" position={Position.Top} id="top" className={`${styles.handle} ${styles.handleTop}`} />
      <Handle type="target" position={Position.Left} id="left" className={`${styles.handle} ${styles.handleLeft}`} />
      <Handle type="target" position={Position.Right} id="right" className={`${styles.handle} ${styles.handleRight}`} />
      <span>End</span>
    </div>
  );
}

function ToolNode({ data }: { data: ToolNodeData }) {
  // MCP Logo SVG - plug icon representing Model Context Protocol
  const MCPIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="#10b981"/>
      <circle cx="12" cy="12" r="3" fill="#10b981"/>
      <path d="M12 6v2M12 16v2M6 12H4M20 12h-2M7.76 7.76l1.41 1.41M14.83 14.83l1.41 1.41M7.76 16.24l1.41-1.41M14.83 9.17l1.41-1.41" stroke="#10b981" strokeWidth="1.5" strokeLinecap="round"/>
    </svg>
  );

  // Microsoft icon for Microsoft Learn servers
  const isMicrosoftServer = data.serverName?.toLowerCase().includes("microsoft") ||
                            data.serverName?.toLowerCase().includes("learn");

  const MicrosoftIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="2" y="2" width="9" height="9" fill="#f25022"/>
      <rect x="13" y="2" width="9" height="9" fill="#7fba00"/>
      <rect x="2" y="13" width="9" height="9" fill="#00a4ef"/>
      <rect x="13" y="13" width="9" height="9" fill="#ffb900"/>
    </svg>
  );

  return (
    <div className={styles.toolNode}>
      <Handle type="source" position={Position.Right} id="right" className={`${styles.handle} ${styles.handleRight}`} />
      <Handle type="source" position={Position.Bottom} id="bottom" className={`${styles.handle} ${styles.handleBottom}`} />
      <div className={styles.toolNodeIcon}>
        {isMicrosoftServer ? <MicrosoftIcon /> : <MCPIcon />}
      </div>
      <div className={styles.toolNodeContent}>
        <div className={styles.toolNodeName}>{data.label}</div>
        <div className={styles.toolNodeTransport}>{data.transport}</div>
      </div>
    </div>
  );
}

interface EdgeConfigModalProps {
  edge: Edge | null;
  onClose: () => void;
  onSave: (edgeId: string, config: EdgeConfig) => void;
}

function EdgeConfigModal({ edge, onClose, onSave }: EdgeConfigModalProps) {
  const existingConfig = (edge?.data as unknown as EdgeConfig) || {} as Partial<EdgeConfig>;

  const [edgeType, setEdgeType] = useState<EdgeType>(existingConfig.edgeType || "default");
  const [label, setLabel] = useState(existingConfig.label || "");
  const [conditionType, setConditionType] = useState<ConditionType>(
    existingConfig.condition?.type || "contains"
  );
  const [conditionValue, setConditionValue] = useState(existingConfig.condition?.value || "");
  const [conditionField, setConditionField] = useState<ConditionField>(
    existingConfig.condition?.field || "last_message"
  );
  const [maxIterations, setMaxIterations] = useState(existingConfig.maxIterations || 3);
  const [priority, setPriority] = useState(existingConfig.priority || 0);

  if (!edge) return null;

  const handleSave = () => {
    const config: EdgeConfig = {
      edgeType,
      label: label || undefined,
      priority,
    };

    if (edgeType === "conditional" || edgeType === "feedback") {
      config.condition = {
        type: conditionType,
        value: conditionValue,
        field: conditionField,
      };
    }

    if (edgeType === "feedback") {
      config.maxIterations = maxIterations;
    }

    onSave(edge.id, config);
    onClose();
  };

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h3>Configure Edge</h3>
          <button className={styles.modalClose} onClick={onClose}>
            ×
          </button>
        </div>

        <div className={styles.modalBody}>
          <div className={styles.formGroup}>
            <label>Edge Type</label>
            <div className={styles.edgeTypeButtons}>
              <button
                className={`${styles.edgeTypeBtn} ${edgeType === "default" ? styles.edgeTypeBtnActive : ""}`}
                onClick={() => setEdgeType("default")}
              >
                <span className={styles.edgeTypeIcon}>→</span>
                Default
              </button>
              <button
                className={`${styles.edgeTypeBtn} ${edgeType === "conditional" ? styles.edgeTypeBtnActive : ""} ${styles.conditionalBtn}`}
                onClick={() => setEdgeType("conditional")}
              >
                <span className={styles.edgeTypeIcon}>⇢</span>
                Conditional
              </button>
              <button
                className={`${styles.edgeTypeBtn} ${edgeType === "feedback" ? styles.edgeTypeBtnActive : ""} ${styles.feedbackBtn}`}
                onClick={() => setEdgeType("feedback")}
              >
                <span className={styles.edgeTypeIcon}>↻</span>
                Feedback
              </button>
            </div>
          </div>

          <div className={styles.formGroup}>
            <label>Label (optional)</label>
            <input
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="e.g., APPROVED, NEEDS_REVISION"
              className={styles.input}
            />
          </div>

          {(edgeType === "conditional" || edgeType === "feedback") && (
            <>
              <div className={styles.formGroup}>
                <label>Condition Field</label>
                <select
                  value={conditionField}
                  onChange={(e) => setConditionField(e.target.value as ConditionField)}
                  className={styles.select}
                >
                  <option value="last_message">Last Message</option>
                  <option value="context">Context</option>
                </select>
              </div>

              <div className={styles.formGroup}>
                <label>Condition Type</label>
                <select
                  value={conditionType}
                  onChange={(e) => setConditionType(e.target.value as ConditionType)}
                  className={styles.select}
                >
                  <option value="contains">Contains</option>
                  <option value="not_contains">Does Not Contain</option>
                  <option value="equals">Equals</option>
                  <option value="regex">Regex Match</option>
                </select>
              </div>

              <div className={styles.formGroup}>
                <label>Condition Value</label>
                <input
                  type="text"
                  value={conditionValue}
                  onChange={(e) => setConditionValue(e.target.value)}
                  placeholder={
                    conditionType === "regex"
                      ? "e.g., (fail|error|issue)"
                      : "e.g., APPROVED"
                  }
                  className={styles.input}
                />
              </div>
            </>
          )}

          {edgeType === "feedback" && (
            <div className={styles.formGroup}>
              <label>Max Iterations</label>
              <input
                type="number"
                value={maxIterations}
                onChange={(e) => setMaxIterations(Math.max(1, parseInt(e.target.value) || 1))}
                min={1}
                max={10}
                className={styles.input}
              />
              <span className={styles.hint}>Prevents infinite loops (1-10)</span>
            </div>
          )}

          <div className={styles.formGroup}>
            <label>Priority</label>
            <input
              type="number"
              value={priority}
              onChange={(e) => setPriority(parseInt(e.target.value) || 0)}
              className={styles.input}
            />
            <span className={styles.hint}>Higher priority edges are evaluated first</span>
          </div>
        </div>

        <div className={styles.modalFooter}>
          <button className={styles.cancelBtn} onClick={onClose}>
            Cancel
          </button>
          <button className={styles.saveConfigBtn} onClick={handleSave}>
            Save Configuration
          </button>
        </div>
      </div>
    </div>
  );
}

const nodeTypes = {
  agent: AgentNode,
  team: TeamNode,
  start: StartNode,
  end: EndNode,
  tool: ToolNode,
};

function normalizeId(name: string): string {
  return name.toLowerCase().replace(/\s+/g, "_");
}

function PipelineEditorInner({ agents, initialNodes, initialEdges, onSave, onPipelineSelect }: PipelineEditorProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, fitView } = useReactFlow();

  const [savedPipelines, setSavedPipelines] = useState<Pipeline[]>([]);
  const [teams, setTeams] = useState<Team[]>([]);
  const [agentConfigs, setAgentConfigs] = useState<AgentConfig[]>([]);
  const [mcpServers, setMcpServers] = useState<MCPServer[]>([]);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"agents" | "pipelines" | "teams" | "tools">("pipelines");
  const [selectedEdge, setSelectedEdge] = useState<Edge | null>(null);

  const defaultNodes: Node[] = useMemo(() => [
    {
      id: "start",
      type: "start",
      position: { x: 250, y: 0 },
      data: {},
      deletable: false,
    },
    {
      id: "end",
      type: "end",
      position: { x: 250, y: 400 },
      data: {},
      deletable: false,
    },
  ], []);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes || defaultNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges || []);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [pipelines, teamsData, agentConfigsData, mcpServersData] = await Promise.all([
        fetchPipelines(),
        fetchTeams(),
        fetchAgentConfigs(),
        fetchMCPServers(true), // Only active servers
      ]);
      setSavedPipelines(pipelines);
      setTeams(teamsData);
      setAgentConfigs(agentConfigsData.filter((a) => a.is_active));
      setMcpServers(mcpServersData);
    } catch (err) {
      console.error("Failed to load data:", err);
    }
  };

  const loadPipeline = useCallback((pipeline: Pipeline) => {
    setSelectedPipelineId(pipeline.id);
    const pipelineNodes = (pipeline.nodes || []).map((n) => ({
      ...n,
      id: String(n.id),
      type: String(n.type || "agent"),
      position: n.position as { x: number; y: number },
      data: n.data as Record<string, unknown>,
    })) as Node[];
    const pipelineEdges = (pipeline.edges || []).map((e) => ({
      ...e,
      id: String(e.id || `${e.source}-${e.target}`),
      source: String(e.source),
      target: String(e.target),
    })) as Edge[];

    setNodes(pipelineNodes);
    setEdges(pipelineEdges);
    onPipelineSelect?.(pipeline.id);

    setTimeout(() => fitView({ padding: 0.2 }), 50);
  }, [setNodes, setEdges, fitView, onPipelineSelect]);

  const handleDeletePipeline = async (pipelineId: string) => {
    if (!confirm("Delete this pipeline?")) return;
    try {
      await deletePipeline(pipelineId);
      setSavedPipelines((prev) => prev.filter((p) => p.id !== pipelineId));
      if (selectedPipelineId === pipelineId) {
        setSelectedPipelineId(null);
        setNodes(defaultNodes);
        setEdges([]);
        onPipelineSelect?.(null);
      }
    } catch (err) {
      console.error("Failed to delete pipeline:", err);
    }
  };

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            type: "smoothstep",
            animated: true,
            style: { stroke: "#f97316", strokeWidth: 2 },
          },
          eds
        )
      );
    },
    [setEdges]
  );

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      const agentData = event.dataTransfer.getData("application/agent");
      const teamData = event.dataTransfer.getData("application/team");
      const toolData = event.dataTransfer.getData("application/tool");

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      // Handle MCP tool drops
      if (toolData) {
        const server: MCPServer = JSON.parse(toolData);
        const toolNodeId = `tool_${server.id}`;

        // Check if tool node already exists
        if (nodes.find((n) => n.id === toolNodeId)) return;

        const newToolNode: Node = {
          id: toolNodeId,
          type: "tool",
          position,
          data: {
            label: server.name,
            mcpServerId: server.id,
            serverName: server.name,
            transport: server.transport,
            description: server.description,
          },
        };

        setNodes((nds) => [...nds, newToolNode]);
        return;
      }

      if (teamData) {
        const team: Team = JSON.parse(teamData);
        const teamAgentIds = team.agent_ids || [];
        const teamNodeId = `team_${team.id}`;

        // Check if team node already exists
        if (nodes.find((n) => n.id === teamNodeId)) return;

        const teamAgents = teamAgentIds.map((agentId) => {
          const agentConfig = agentConfigs.find((a) => a.id === agentId);
          return {
            id: agentId,
            name: agentConfig?.name || agentId,
            description: agentConfig?.description || "",
          };
        });

        // Create single team node
        const newTeamNode: Node = {
          id: teamNodeId,
          type: "team",
          position,
          data: {
            teamId: team.id,
            teamName: team.name,
            agentIds: teamAgentIds,
            leadAgentId: team.lead_agent_id,
            agents: teamAgents,
          },
        };

        setNodes((nds) => [...nds, newTeamNode]);
        return;
      }

      if (agentData) {
        const agent: AgentInfo & { id?: string } = JSON.parse(agentData);
        const agentId = agent.id || normalizeId(agent.name);

        const existingNode = nodes.find((n) => n.id === agentId);
        if (existingNode) return;

        const newNode: Node = {
          id: agentId,
          type: "agent",
          position,
          data: {
            label: agent.name,
            description: agent.description,
            agentId: agentId,
            agentName: agentId,
            name: agentId,
          },
        };

        setNodes((nds) => [...nds, newNode]);
      }
    },
    [nodes, agents, agentConfigs, screenToFlowPosition, setNodes]
  );

  const handleSave = useCallback(async () => {
    setIsLoading(true);
    try {
      // If we have a selected pipeline, update it instead of creating new
      if (selectedPipelineId) {
        const currentPipeline = savedPipelines.find((p) => p.id === selectedPipelineId);
        const pipeline = await updatePipeline(selectedPipelineId, {
          name: currentPipeline?.name || "Updated Pipeline",
          description: currentPipeline?.description || "Custom pipeline",
          nodes: nodes.map((n) => ({ ...n })),
          edges: edges.map((e) => ({ ...e })),
        });
        setSavedPipelines((prev) =>
          prev.map((p) => (p.id === selectedPipelineId ? pipeline : p))
        );
      } else {
        // No pipeline selected, create new one
        const name = prompt("Pipeline name:");
        if (!name) {
          setIsLoading(false);
          return;
        }
        const pipeline = await createPipeline({
          name,
          description: "Custom pipeline",
          nodes: nodes.map((n) => ({ ...n })),
          edges: edges.map((e) => ({ ...e })),
        });
        setSavedPipelines((prev) => [...prev, pipeline]);
        setSelectedPipelineId(pipeline.id);
        onPipelineSelect?.(pipeline.id);
      }
    } catch (err) {
      console.error("Failed to save pipeline:", err);
      alert("Failed to save pipeline");
    } finally {
      setIsLoading(false);
    }
  }, [nodes, edges, selectedPipelineId, savedPipelines, onPipelineSelect]);

  const handleClear = useCallback(() => {
    setSelectedPipelineId(null);
    setNodes(defaultNodes);
    setEdges([]);
    onPipelineSelect?.(null);
  }, [defaultNodes, setNodes, setEdges, onPipelineSelect]);

  const onEdgeClick = useCallback((_: React.MouseEvent, edge: Edge) => {
    setSelectedEdge(edge);
  }, []);

  const handleEdgeConfigSave = useCallback((edgeId: string, config: EdgeConfig) => {
    setEdges((eds) =>
      eds.map((e) => {
        if (e.id !== edgeId) return e;

        // Apply styling based on edge type
        let style = { stroke: "#f97316", strokeWidth: 2 };
        let animated = true;
        let labelStyle = {};

        if (config.edgeType === "conditional") {
          style = { stroke: "#3b82f6", strokeWidth: 2 };
          animated = false;
        } else if (config.edgeType === "feedback") {
          style = { stroke: "#ef4444", strokeWidth: 2 };
          animated = true;
        }

        return {
          ...e,
          data: config,
          style,
          animated,
          label: config.label,
          labelStyle: {
            fill: "#fff",
            fontSize: 11,
            fontWeight: 500,
          },
          labelBgStyle: {
            fill: config.edgeType === "feedback" ? "#ef4444" : config.edgeType === "conditional" ? "#3b82f6" : "#f97316",
            fillOpacity: 0.9,
          },
          labelBgPadding: [4, 8] as [number, number],
          labelBgBorderRadius: 4,
        };
      })
    );
  }, [setEdges]);

  const getEdgeStyle = useCallback((edge: Edge) => {
    const config = edge.data as EdgeConfig | undefined;
    if (!config) return { stroke: "#f97316", strokeWidth: 2 };

    if (config.edgeType === "conditional") {
      return { stroke: "#3b82f6", strokeWidth: 2, strokeDasharray: "5,5" };
    } else if (config.edgeType === "feedback") {
      return { stroke: "#ef4444", strokeWidth: 2, strokeDasharray: "5,5" };
    }
    return { stroke: "#f97316", strokeWidth: 2 };
  }, []);

  return (
    <div className={styles.editorContainer}>
      <div className={styles.sidebar}>
        <div className={styles.tabBar}>
          <button
            className={`${styles.tab} ${activeTab === "pipelines" ? styles.tabActive : ""}`}
            onClick={() => setActiveTab("pipelines")}
          >
            Pipelines
          </button>
          <button
            className={`${styles.tab} ${activeTab === "agents" ? styles.tabActive : ""}`}
            onClick={() => setActiveTab("agents")}
          >
            Agents
          </button>
          <button
            className={`${styles.tab} ${activeTab === "teams" ? styles.tabActive : ""}`}
            onClick={() => setActiveTab("teams")}
          >
            Teams
          </button>
          <button
            className={`${styles.tab} ${activeTab === "tools" ? styles.tabActive : ""}`}
            onClick={() => setActiveTab("tools")}
          >
            Tools
          </button>
        </div>

        <div className={styles.sidebarContent}>
          {activeTab === "pipelines" && (
            <>
              {savedPipelines.length === 0 ? (
                <div className={styles.emptyState}>No saved pipelines</div>
              ) : (
                savedPipelines.map((pipeline) => (
                  <div
                    key={pipeline.id}
                    className={`${styles.pipelineItem} ${selectedPipelineId === pipeline.id ? styles.pipelineSelected : ""}`}
                  >
                    <div
                      className={styles.pipelineInfo}
                      onClick={() => loadPipeline(pipeline)}
                    >
                      <div className={styles.pipelineName}>{pipeline.name}</div>
                      <div className={styles.pipelineMeta}>
                        {(pipeline.nodes || []).filter((n) => (n as { type?: string }).type === "agent").length} agents
                      </div>
                    </div>
                    <button
                      className={styles.deleteBtn}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeletePipeline(pipeline.id);
                      }}
                      title="Delete pipeline"
                    >
                      ×
                    </button>
                  </div>
                ))
              )}
            </>
          )}

          {activeTab === "agents" && (
            <>
              {agentConfigs.map((agent) => {
                const isUsed = nodes.some((n) => n.id === agent.id);
                return (
                  <div
                    key={agent.id}
                    className={`${styles.agentItem} ${isUsed ? styles.agentUsed : ""}`}
                    draggable={!isUsed}
                    onDragStart={(e) => {
                      if (isUsed) {
                        e.preventDefault();
                        return;
                      }
                      e.dataTransfer.setData("application/agent", JSON.stringify({ id: agent.id, name: agent.name, description: agent.description }));
                      e.dataTransfer.effectAllowed = "move";
                    }}
                  >
                    <div className={styles.agentItemName}>{agent.name}</div>
                    <div className={styles.agentItemDesc}>{agent.description}</div>
                    {isUsed && <span className={styles.usedBadge}>In use</span>}
                  </div>
                );
              })}
            </>
          )}

          {activeTab === "teams" && (
            <>
              {teams.length === 0 ? (
                <div className={styles.emptyState}>No teams created</div>
              ) : (
                teams.map((team) => {
                  const teamAgentIds = team.agent_ids || [];
                  const teamNodeId = `team_${team.id}`;
                  const isTeamUsed = nodes.some((n) => n.id === teamNodeId);
                  return (
                    <div
                      key={team.id}
                      className={`${styles.teamItem} ${isTeamUsed ? styles.agentUsed : ""}`}
                      draggable={!isTeamUsed}
                      onDragStart={(e) => {
                        if (isTeamUsed) {
                          e.preventDefault();
                          return;
                        }
                        e.dataTransfer.setData("application/team", JSON.stringify(team));
                        e.dataTransfer.effectAllowed = "move";
                      }}
                    >
                      <div className={styles.teamItemHeader}>
                        <div className={styles.agentItemName}>{team.name}</div>
                        {isTeamUsed && <span className={styles.usedBadge}>In use</span>}
                      </div>
                      <div className={styles.teamItemAgents}>
                        {teamAgentIds.slice(0, 3).map((agentId) => {
                          const agentConfig = agentConfigs.find((a) => a.id === agentId);
                          return (
                            <span key={agentId} className={styles.teamItemAgent}>
                              {agentConfig?.name || agentId}
                              {agentId === team.lead_agent_id && " ★"}
                            </span>
                          );
                        })}
                        {teamAgentIds.length > 3 && (
                          <span className={styles.teamItemMore}>+{teamAgentIds.length - 3} more</span>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </>
          )}

          {activeTab === "tools" && (
            <>
              {mcpServers.length === 0 ? (
                <div className={styles.emptyState}>No MCP servers configured</div>
              ) : (
                mcpServers.map((server) => {
                  const toolNodeId = `tool_${server.id}`;
                  const isUsed = nodes.some((n) => n.id === toolNodeId);
                  const isMicrosoft = server.name.toLowerCase().includes("microsoft") ||
                                     server.name.toLowerCase().includes("learn");
                  return (
                    <div
                      key={server.id}
                      className={`${styles.toolItem} ${isUsed ? styles.agentUsed : ""}`}
                      draggable={!isUsed}
                      onDragStart={(e) => {
                        if (isUsed) {
                          e.preventDefault();
                          return;
                        }
                        e.dataTransfer.setData("application/tool", JSON.stringify(server));
                        e.dataTransfer.effectAllowed = "move";
                      }}
                    >
                      <div className={styles.toolItemHeader}>
                        <div className={styles.toolItemIcon}>
                          {isMicrosoft ? (
                            <svg width="16" height="16" viewBox="0 0 24 24">
                              <rect x="2" y="2" width="9" height="9" fill="#f25022"/>
                              <rect x="13" y="2" width="9" height="9" fill="#7fba00"/>
                              <rect x="2" y="13" width="9" height="9" fill="#00a4ef"/>
                              <rect x="13" y="13" width="9" height="9" fill="#ffb900"/>
                            </svg>
                          ) : (
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="#10b981">
                              <circle cx="12" cy="12" r="10"/>
                              <circle cx="12" cy="12" r="4" fill="#181818"/>
                            </svg>
                          )}
                        </div>
                        <div className={styles.toolItemName}>{server.name}</div>
                        {isUsed && <span className={styles.usedBadge}>In use</span>}
                      </div>
                      <div className={styles.toolItemMeta}>
                        <span className={styles.toolItemTransport}>{server.transport}</span>
                        {server.description && (
                          <span className={styles.toolItemDesc}>{server.description}</span>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </>
          )}
        </div>

        <div className={styles.sidebarActions}>
          <button className={styles.saveBtn} onClick={handleSave} disabled={isLoading}>
            {isLoading ? "Saving..." : "Save Pipeline"}
          </button>
          <button className={styles.clearBtn} onClick={handleClear}>
            New Pipeline
          </button>
        </div>
      </div>

      <div className={styles.canvas} ref={reactFlowWrapper}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onEdgeClick={onEdgeClick}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
          deleteKeyCode={["Backspace", "Delete"]}
          snapToGrid
          snapGrid={[20, 20]}
        >
          <Background color="#333" gap={20} />
          <Controls showInteractive={false} />
          <MiniMap
            nodeColor={(node) => {
              if (node.type === "start") return "#22c55e";
              if (node.type === "end") return "#ef4444";
              if (node.type === "team") return "#a855f7";
              if (node.type === "tool") return "#10b981";
              return "#f97316";
            }}
            maskColor="rgba(0, 0, 0, 0.8)"
          />
        </ReactFlow>
      </div>

      {selectedEdge && (
        <EdgeConfigModal
          edge={selectedEdge}
          onClose={() => setSelectedEdge(null)}
          onSave={handleEdgeConfigSave}
        />
      )}
    </div>
  );
}

export default function PipelineEditor(props: PipelineEditorProps) {
  return (
    <ReactFlowProvider>
      <PipelineEditorInner {...props} />
    </ReactFlowProvider>
  );
}
