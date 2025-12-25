"use client";

import { useState, useEffect, useCallback } from "react";
import {
  fetchAgentConfigs,
  updateAgent,
  createAgent,
  deleteAgent,
  fetchAvailableTools,
  type AgentConfig,
} from "@/lib/api";
import styles from "./AgentEditor.module.css";

interface EditingAgent {
  id: string;
  name: string;
  description: string;
  system_prompt: string;
  capabilities: string[];
  enabled_tools: string[];
  model_override: string;
  is_active: boolean;
  is_builtin: boolean;
  isNew?: boolean;
}

export default function AgentEditor() {
  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [tools, setTools] = useState<Array<{ id: string; name: string; description: string }>>([]);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [editingAgent, setEditingAgent] = useState<EditingAgent | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      setIsLoading(true);
      const [agentData, toolData] = await Promise.all([
        fetchAgentConfigs(),
        fetchAvailableTools(),
      ]);
      setAgents(agentData);
      setTools(toolData);
    } catch (err) {
      setError("Failed to load agents");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleSelectAgent = (agentId: string) => {
    const agent = agents.find((a) => a.id === agentId);
    if (agent) {
      setSelectedAgent(agentId);
      setEditingAgent({
        ...agent,
        model_override: agent.model_override || "",
      });
      setError(null);
      setSuccess(null);
    }
  };

  // Generate ID from name: "My Agent Name" -> "my_agent_name"
  const generateIdFromName = (name: string): string => {
    return name
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s]/g, "") // Remove special chars
      .replace(/\s+/g, "_"); // Replace spaces with underscores
  };

  const handleNewAgent = () => {
    setSelectedAgent(null);
    setEditingAgent({
      id: "",
      name: "",
      description: "",
      system_prompt: "",
      capabilities: [],
      enabled_tools: [],
      model_override: "",
      is_active: true,
      is_builtin: false,
      isNew: true,
    });
    setError(null);
    setSuccess(null);
  };

  const handleNameChange = (name: string) => {
    if (!editingAgent) return;
    const newState: EditingAgent = { ...editingAgent, name };
    // Auto-generate ID for new agents
    if (editingAgent.isNew) {
      newState.id = generateIdFromName(name);
    }
    setEditingAgent(newState);
  };

  const handleSave = async () => {
    if (!editingAgent) return;

    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      if (editingAgent.isNew) {
        // Create new agent
        if (!editingAgent.id || !editingAgent.name) {
          throw new Error("ID and Name are required");
        }
        await createAgent({
          id: editingAgent.id,
          name: editingAgent.name,
          description: editingAgent.description,
          system_prompt: editingAgent.system_prompt,
          capabilities: editingAgent.capabilities,
          enabled_tools: editingAgent.enabled_tools,
          model_override: editingAgent.model_override || undefined,
        });
        setSuccess("Agent created successfully");
      } else {
        // Update existing agent
        await updateAgent(editingAgent.id, {
          name: editingAgent.name,
          description: editingAgent.description,
          system_prompt: editingAgent.system_prompt,
          capabilities: editingAgent.capabilities,
          enabled_tools: editingAgent.enabled_tools,
          model_override: editingAgent.model_override || undefined,
          is_active: editingAgent.is_active,
        });
        setSuccess("Agent updated successfully");
      }

      await loadData();
      if (editingAgent.isNew) {
        setSelectedAgent(editingAgent.id);
        setEditingAgent({ ...editingAgent, isNew: false });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save agent");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!editingAgent || editingAgent.isNew) return;

    const confirmMsg = editingAgent.is_builtin
      ? `Reset "${editingAgent.name}" to defaults?`
      : `Delete "${editingAgent.name}"? This cannot be undone.`;

    if (!confirm(confirmMsg)) return;

    try {
      await deleteAgent(editingAgent.id);
      setSuccess(editingAgent.is_builtin ? "Agent reset to defaults" : "Agent deleted");
      setSelectedAgent(null);
      setEditingAgent(null);
      await loadData();
    } catch (err) {
      setError("Failed to delete agent");
    }
  };

  const handleToggleTool = (toolId: string) => {
    if (!editingAgent) return;
    const tools = editingAgent.enabled_tools.includes(toolId)
      ? editingAgent.enabled_tools.filter((t) => t !== toolId)
      : [...editingAgent.enabled_tools, toolId];
    setEditingAgent({ ...editingAgent, enabled_tools: tools });
  };

  const handleCapabilitiesChange = (value: string) => {
    if (!editingAgent) return;
    const caps = value.split("\n").filter((c) => c.trim());
    setEditingAgent({ ...editingAgent, capabilities: caps });
  };

  if (isLoading) {
    return (
      <div className={styles.panel}>
        <div className={styles.placeholder}>Loading agents...</div>
      </div>
    );
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span>Agent Configuration</span>
        <button onClick={handleNewAgent} className={styles.addBtn}>
          + New Agent
        </button>
      </div>

      <div className={styles.layout}>
        {/* Agent List */}
        <div className={styles.agentList}>
          {agents.map((agent) => (
            <div
              key={agent.id}
              className={`${styles.agentItem} ${selectedAgent === agent.id ? styles.selected : ""} ${!agent.is_active ? styles.inactive : ""}`}
              onClick={() => handleSelectAgent(agent.id)}
            >
              <div className={styles.agentName}>
                {agent.name}
                {agent.is_builtin && <span className={styles.builtin}>builtin</span>}
              </div>
              <div className={styles.agentRole}>{agent.role}</div>
            </div>
          ))}
        </div>

        {/* Editor Panel */}
        <div className={styles.editor}>
          {editingAgent ? (
            <>
              {error && <div className={styles.error}>{error}</div>}
              {success && <div className={styles.success}>{success}</div>}

              {/* Show ID field only for existing agents (read-only) */}
              {!editingAgent.isNew && (
                <div className={styles.field}>
                  <label>Agent ID</label>
                  <input
                    type="text"
                    value={editingAgent.id}
                    disabled
                    className={styles.input}
                  />
                </div>
              )}

              <div className={styles.field}>
                <label>Name</label>
                <input
                  type="text"
                  value={editingAgent.name}
                  onChange={(e) => handleNameChange(e.target.value)}
                  placeholder="My Custom Agent"
                  className={styles.input}
                />
                {editingAgent.isNew && editingAgent.id && (
                  <span className={styles.hint}>ID will be: {editingAgent.id}</span>
                )}
              </div>

              <div className={styles.field}>
                <label>Description</label>
                <input
                  type="text"
                  value={editingAgent.description}
                  onChange={(e) =>
                    setEditingAgent({ ...editingAgent, description: e.target.value })
                  }
                  placeholder="What this agent does..."
                  className={styles.input}
                />
              </div>

              <div className={styles.field}>
                <label>System Prompt</label>
                <textarea
                  value={editingAgent.system_prompt}
                  onChange={(e) =>
                    setEditingAgent({ ...editingAgent, system_prompt: e.target.value })
                  }
                  placeholder="You are a helpful assistant that..."
                  className={styles.textarea}
                  rows={6}
                />
              </div>

              <div className={styles.field}>
                <label>Capabilities (one per line)</label>
                <textarea
                  value={editingAgent.capabilities.join("\n")}
                  onChange={(e) => handleCapabilitiesChange(e.target.value)}
                  placeholder="Write code&#10;Debug issues&#10;Explain concepts"
                  className={styles.textarea}
                  rows={3}
                />
              </div>

              <div className={styles.field}>
                <label>Model Override (optional)</label>
                <input
                  type="text"
                  value={editingAgent.model_override}
                  onChange={(e) =>
                    setEditingAgent({ ...editingAgent, model_override: e.target.value })
                  }
                  placeholder="Leave empty for system default"
                  className={styles.input}
                />
              </div>

              <div className={styles.field}>
                <label>Enabled Tools</label>
                <div className={styles.toolGrid}>
                  {tools.map((tool) => (
                    <label key={tool.id} className={styles.toolCheck}>
                      <input
                        type="checkbox"
                        checked={editingAgent.enabled_tools.includes(tool.id)}
                        onChange={() => handleToggleTool(tool.id)}
                      />
                      <span>{tool.name}</span>
                    </label>
                  ))}
                </div>
              </div>

              {!editingAgent.isNew && (
                <div className={styles.field}>
                  <label className={styles.toggleLabel}>
                    <input
                      type="checkbox"
                      checked={editingAgent.is_active}
                      onChange={(e) =>
                        setEditingAgent({ ...editingAgent, is_active: e.target.checked })
                      }
                    />
                    <span>Agent Active</span>
                  </label>
                </div>
              )}

              <div className={styles.actions}>
                <button
                  onClick={handleSave}
                  disabled={isSaving}
                  className={styles.saveBtn}
                >
                  {isSaving ? "Saving..." : editingAgent.isNew ? "Create Agent" : "Save Changes"}
                </button>
                {!editingAgent.isNew && (
                  <button onClick={handleDelete} className={styles.deleteBtn}>
                    {editingAgent.is_builtin ? "Reset to Defaults" : "Delete Agent"}
                  </button>
                )}
              </div>
            </>
          ) : (
            <div className={styles.placeholder}>
              Select an agent to edit or create a new one
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
