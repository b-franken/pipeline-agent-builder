"use client";

import { useEffect, useMemo } from "react";
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  Position,
  MarkerType,
  Handle,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import styles from "./ExecutionGraph.module.css";

export interface ExecutionEvent {
  type: "agent_start" | "agent_end" | "handoff" | "tool_call" | "task_start" | "task_end" | "error" | "finish";
  agent?: string;
  from?: string;
  to?: string;
  tool?: string;
  message?: string;
  timestamp?: string;
}

export interface AgentInfo {
  name: string;
  description: string;
  status: "idle" | "active" | "completed" | "error";
}

interface PipelineNode {
  id?: unknown;
  type?: unknown;
  position?: unknown;
  data?: unknown;
}

interface PipelineEdge {
  id?: unknown;
  source?: unknown;
  target?: unknown;
  data?: unknown;
}

interface ExecutionGraphProps {
  agents: AgentInfo[];
  events: ExecutionEvent[];
  isExecuting: boolean;
  pipelineNodes?: PipelineNode[];
  pipelineEdges?: PipelineEdge[];
}

interface AgentNodeData {
  label: string;
  description?: string;
  status: string;
}

interface SupervisorNodeData {
  status: string;
}

interface StartNodeData {
  label?: string;
}

interface EndNodeData {
  label?: string;
  reached?: boolean;
}

interface TeamNodeData {
  label?: string;
  teamName?: string;
  agents?: Array<{ id: string; name: string }>;
  status: string;
}

interface ToolNodeData {
  label: string;
  serverName?: string;
  transport?: string;
  status: string;
}

function normalizeId(name: string): string {
  return name.toLowerCase().replace(/\s+/g, "_");
}

function AgentNode({ data }: { data: AgentNodeData }) {
  const statusClass = data.status === "active"
    ? styles.nodeActive
    : data.status === "completed"
    ? styles.nodeCompleted
    : data.status === "error"
    ? styles.nodeError
    : styles.nodeIdle;

  return (
    <div className={`${styles.agentNode} ${statusClass}`}>
      <Handle type="target" position={Position.Top} className={styles.handle} />
      <div className={styles.nodeHeader}>
        <span className={styles.nodeName}>{data.label}</span>
      </div>
      {data.description && (
        <div className={styles.nodeDescription}>{data.description}</div>
      )}
      {data.status === "active" && (
        <div className={styles.nodeStatus}>
          <span className={styles.pulse} />
          Processing
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className={styles.handle} />
    </div>
  );
}

function SupervisorNode({ data }: { data: SupervisorNodeData }) {
  const statusClass = data.status === "active"
    ? styles.nodeActive
    : data.status === "completed"
    ? styles.nodeCompleted
    : styles.nodeIdle;

  return (
    <div className={`${styles.supervisorNode} ${statusClass}`}>
      <Handle type="target" position={Position.Top} className={styles.handle} />
      <div className={styles.nodeHeader}>
        <span className={styles.nodeName}>Supervisor</span>
      </div>
      <div className={styles.nodeDescription}>Routes tasks</div>
      {data.status === "active" && (
        <div className={styles.nodeStatus}>
          <span className={styles.pulse} />
          Routing
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className={styles.handle} />
    </div>
  );
}

function StartNode({ data }: { data: StartNodeData }) {
  return (
    <div className={styles.startNode}>
      <span>{data.label || "Start"}</span>
      <Handle type="source" position={Position.Bottom} className={styles.handle} />
    </div>
  );
}

function EndNode({ data }: { data: EndNodeData }) {
  return (
    <div className={`${styles.endNode} ${data.reached ? styles.endReached : ""}`}>
      <Handle type="target" position={Position.Top} className={styles.handle} />
      <span>{data.label || "End"}</span>
    </div>
  );
}

function TeamNode({ data }: { data: TeamNodeData }) {
  const statusClass = data.status === "active"
    ? styles.nodeActive
    : data.status === "completed"
    ? styles.nodeCompleted
    : data.status === "error"
    ? styles.nodeError
    : styles.nodeIdle;

  return (
    <div className={`${styles.orgChart} ${statusClass}`}>
      <div className={styles.orgHeader}>
        <Handle type="target" position={Position.Top} className={styles.handle} />
        <span className={styles.orgTitle}>{data.teamName || data.label || "Team"}</span>
        {data.status === "active" && <span className={styles.pulse} />}
      </div>
      {data.agents && data.agents.length > 0 && (
        <>
          <div className={styles.orgLine} />
          <div className={styles.orgBranches}>
            {data.agents.map((agent) => (
              <div key={agent.id} className={styles.orgBranch}>
                <div className={styles.orgBranchLine} />
                <div className={styles.orgAgent}>
                  <span className={styles.orgAgentName}>{agent.name}</span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
      <Handle type="source" position={Position.Bottom} className={styles.handle} />
    </div>
  );
}

function ToolNode({ data }: { data: ToolNodeData }) {
  const isMicrosoft = data.serverName?.toLowerCase().includes("microsoft") ||
                      data.serverName?.toLowerCase().includes("learn") ||
                      data.label?.toLowerCase().includes("microsoft") ||
                      data.label?.toLowerCase().includes("learn");

  const MicrosoftIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <rect x="2" y="2" width="9" height="9" fill="#f25022"/>
      <rect x="13" y="2" width="9" height="9" fill="#7fba00"/>
      <rect x="2" y="13" width="9" height="9" fill="#00a4ef"/>
      <rect x="13" y="13" width="9" height="9" fill="#ffb900"/>
    </svg>
  );

  const MCPIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="#10b981">
      <circle cx="12" cy="12" r="10"/>
      <circle cx="12" cy="12" r="4" fill="#1a1a1a"/>
    </svg>
  );

  return (
    <div className={styles.toolNode}>
      <Handle type="source" position={Position.Right} id="right" className={styles.handle} />
      <Handle type="source" position={Position.Bottom} id="bottom" className={styles.handle} />
      <div className={styles.toolNodeIcon}>
        {isMicrosoft ? <MicrosoftIcon /> : <MCPIcon />}
      </div>
      <div className={styles.toolNodeContent}>
        <div className={styles.toolNodeName}>{data.label}</div>
        {data.transport && (
          <div className={styles.toolNodeTransport}>{data.transport}</div>
        )}
      </div>
    </div>
  );
}

const nodeTypes = {
  agent: AgentNode,
  supervisor: SupervisorNode,
  start: StartNode,
  end: EndNode,
  team: TeamNode,
  tool: ToolNode,
};

export default function ExecutionGraph({ agents, events, isExecuting, pipelineNodes, pipelineEdges }: ExecutionGraphProps) {
  const hasPipeline = pipelineNodes && pipelineNodes.length > 0;

  const initialNodes = useMemo(() => {
    if (hasPipeline && pipelineNodes) {
      return pipelineNodes.map((n) => {
        const nodeType = String(n.type || "agent");
        const nodeData = n.data as {
          label?: string;
          description?: string;
          agentName?: string;
          teamName?: string;
          serverName?: string;
          mcpServerId?: string;
          transport?: string;
          agents?: Array<{ id: string; name: string }>;
        } | undefined;

        if (nodeType === "team") {
          return {
            id: String(n.id || ""),
            type: "team",
            position: (n.position || { x: 0, y: 0 }) as { x: number; y: number },
            data: {
              label: nodeData?.teamName || nodeData?.label || "Team",
              teamName: nodeData?.teamName || nodeData?.label || "Team",
              agents: nodeData?.agents || [],
              status: "idle",
            },
          };
        }

        if (nodeType === "tool") {
          return {
            id: String(n.id || ""),
            type: "tool",
            position: (n.position || { x: 0, y: 0 }) as { x: number; y: number },
            data: {
              label: nodeData?.label || nodeData?.serverName || "MCP Tool",
              serverName: nodeData?.serverName || nodeData?.label || "",
              transport: nodeData?.transport || "stdio",
              status: "idle",
            },
          };
        }

        return {
          id: String(n.id || ""),
          type: nodeType,
          position: (n.position || { x: 0, y: 0 }) as { x: number; y: number },
          data: {
            label: nodeData?.label || nodeData?.agentName || String(n.id || ""),
            description: nodeData?.description || "",
            status: "idle",
            reached: false,
          },
        };
      }) as Node[];
    }

    const nodes: Node[] = [];
    const agentCount = agents.length;
    const spacing = 180;
    const startX = -(agentCount * spacing) / 2 + spacing / 2;

    nodes.push({
      id: "start",
      type: "start",
      position: { x: 0, y: 0 },
      data: { label: "Task" },
    });

    nodes.push({
      id: "supervisor",
      type: "supervisor",
      position: { x: 0, y: 100 },
      data: { status: "idle" },
    });

    agents.forEach((agent, index) => {
      nodes.push({
        id: normalizeId(agent.name),
        type: "agent",
        position: { x: startX + index * spacing, y: 250 },
        data: {
          label: agent.name,
          description: agent.description,
          status: "idle",
        },
      });
    });

    nodes.push({
      id: "end",
      type: "end",
      position: { x: 0, y: 400 },
      data: { label: "Complete", reached: false },
    });

    return nodes;
  }, [agents, hasPipeline, pipelineNodes]);

  const initialEdges = useMemo(() => {
    if (hasPipeline && pipelineEdges) {
      return pipelineEdges.map((e) => {
        const edgeData = e.data as { edgeType?: string; label?: string } | undefined;
        const edgeType = edgeData?.edgeType || "default";

        // Different colors for different edge types
        let strokeColor = "#404040";
        let strokeDasharray: string | undefined = undefined;

        if (edgeType === "feedback") {
          strokeColor = "#f59e0b"; // Amber/orange for feedback loops
          strokeDasharray = "8,4";
        } else if (edgeType === "conditional") {
          strokeColor = "#3b82f6"; // Blue for conditional edges
          strokeDasharray = "5,5";
        }

        return {
          id: String(e.id || `${e.source}-${e.target}`),
          source: String(e.source || ""),
          target: String(e.target || ""),
          type: "smoothstep",
          animated: false,
          style: { stroke: strokeColor, strokeWidth: 2, strokeDasharray },
          markerEnd: { type: MarkerType.ArrowClosed, color: strokeColor },
          label: edgeData?.label,
          labelStyle: edgeData?.label ? { fill: "#fff", fontSize: 10, fontWeight: 500 } : undefined,
          labelBgStyle: edgeData?.label ? { fill: strokeColor, fillOpacity: 0.9 } : undefined,
          labelBgPadding: edgeData?.label ? [4, 6] as [number, number] : undefined,
          labelBgBorderRadius: 4,
          data: edgeData,
        };
      }) as Edge[];
    }

    const edges: Edge[] = [];

    edges.push({
      id: "start-supervisor",
      source: "start",
      target: "supervisor",
      type: "smoothstep",
      animated: false,
      style: { stroke: "#404040", strokeWidth: 2 },
    });

    agents.forEach((agent) => {
      const agentId = normalizeId(agent.name);
      edges.push({
        id: `supervisor-${agentId}`,
        source: "supervisor",
        target: agentId,
        type: "smoothstep",
        animated: false,
        style: { stroke: "#404040", strokeWidth: 2, strokeDasharray: "5,5" },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#404040" },
      });

      edges.push({
        id: `${agentId}-end`,
        source: agentId,
        target: "end",
        type: "smoothstep",
        animated: false,
        style: { stroke: "#404040", strokeWidth: 2, strokeDasharray: "5,5" },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#404040" },
      });
    });

    return edges;
  }, [agents, hasPipeline, pipelineEdges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const executionPath = useMemo(() => {
    const path: string[] = [];

    for (const event of events) {
      if (event.type === "task_start") {
        if (!path.includes("start")) path.push("start");
      } else if (event.type === "agent_start" && event.agent) {
        const agentId = normalizeId(event.agent);
        if (!path.includes(agentId)) path.push(agentId);
      } else if (event.type === "handoff" && event.to) {
        const target = event.to === "__end__" ? "end" : normalizeId(event.to);
        if (!path.includes(target)) path.push(target);
      } else if (event.type === "task_end" || event.type === "finish") {
        if (!path.includes("end")) path.push("end");
      }
    }

    return path;
  }, [events]);

  const activeAgent = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const event = events[i];
      if (event.type === "agent_start" && event.agent) {
        return normalizeId(event.agent);
      }
      if (event.type === "agent_end" || event.type === "task_end" || event.type === "finish") {
        return null;
      }
    }
    return null;
  }, [events]);

  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => {
        const nodeId = node.id;
        let status = "idle";

        if (executionPath.includes(nodeId)) {
          if (nodeId === activeAgent && isExecuting) {
            status = "active";
          } else {
            status = "completed";
          }
        }

        if (nodeId === "end") {
          return {
            ...node,
            data: { ...node.data, reached: executionPath.includes("end") },
          };
        }

        return {
          ...node,
          data: { ...node.data, status },
        };
      })
    );

    setEdges((eds) =>
      eds.map((edge) => {
        const sourceIdx = executionPath.indexOf(edge.source);
        const targetIdx = executionPath.indexOf(edge.target);
        const isOnPath = sourceIdx !== -1 && targetIdx !== -1 && targetIdx === sourceIdx + 1;
        const isActive = isOnPath && isExecuting && targetIdx === executionPath.length - 1;

        // Check edge type for proper coloring
        const edgeData = edge.data as { edgeType?: string } | undefined;
        const edgeType = edgeData?.edgeType || "default";
        const isFeedback = edgeType === "feedback";
        const isConditional = edgeType === "conditional";

        // Determine colors based on edge type and execution state
        let strokeColor: string;
        let strokeDasharray: string | undefined;

        if (isOnPath) {
          strokeColor = "#1ed760"; // Green when active on path
          strokeDasharray = undefined;
        } else if (isFeedback) {
          strokeColor = "#f59e0b"; // Amber/orange for feedback
          strokeDasharray = "8,4";
        } else if (isConditional) {
          strokeColor = "#3b82f6"; // Blue for conditional
          strokeDasharray = "5,5";
        } else {
          strokeColor = "#404040"; // Gray for default
          strokeDasharray = "5,5";
        }

        return {
          ...edge,
          animated: isActive || (isFeedback && isExecuting),
          style: {
            stroke: strokeColor,
            strokeWidth: isOnPath ? 3 : 2,
            strokeDasharray,
          },
          markerEnd: { type: MarkerType.ArrowClosed, color: strokeColor },
        };
      })
    );
  }, [events, executionPath, activeAgent, isExecuting, setNodes, setEdges]);

  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  return (
    <div className={styles.graphContainer}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={true}
        zoomOnScroll={true}
      >
        <Background color="#282828" gap={24} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
