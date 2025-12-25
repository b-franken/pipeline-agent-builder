"use client";

import { useState, useEffect, useCallback } from "react";
import styles from "./TeamEditor.module.css";
import {
  fetchTeams,
  createTeam,
  updateTeam,
  deleteTeam,
  fetchAgentConfigs,
  type Team,
  type AgentConfig,
} from "@/lib/api";

interface TeamEditorProps {
  onTeamSelect?: (team: Team | null) => void;
}

export default function TeamEditor({ onTeamSelect }: TeamEditorProps) {
  const [teams, setTeams] = useState<Team[]>([]);
  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const [editName, setEditName] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [editAgentIds, setEditAgentIds] = useState<string[]>([]);
  const [editLeadAgentId, setEditLeadAgentId] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [teamsData, agentsData] = await Promise.all([
        fetchTeams(),
        fetchAgentConfigs(),
      ]);
      setTeams(teamsData);
      setAgents(agentsData.filter((a) => a.is_active));
    } catch (err) {
      console.error("Failed to load data:", err);
    }
  };

  const selectTeam = useCallback((team: Team | null) => {
    setSelectedTeam(team);
    setIsCreating(false);
    if (team) {
      setEditName(team.name);
      setEditDescription(team.description);
      setEditAgentIds(team.agent_ids || []);
      setEditLeadAgentId(team.lead_agent_id);
    } else {
      resetForm();
    }
    onTeamSelect?.(team);
  }, [onTeamSelect]);

  const resetForm = () => {
    setEditName("");
    setEditDescription("");
    setEditAgentIds([]);
    setEditLeadAgentId(null);
  };

  const startCreating = () => {
    setSelectedTeam(null);
    setIsCreating(true);
    resetForm();
  };

  const handleSave = async () => {
    if (!editName.trim()) return;
    setIsLoading(true);

    try {
      if (isCreating) {
        const newTeam = await createTeam({
          name: editName,
          description: editDescription,
          agent_ids: editAgentIds,
          lead_agent_id: editLeadAgentId || undefined,
        });
        setTeams((prev) => [...prev, newTeam]);
        selectTeam(newTeam);
      } else if (selectedTeam) {
        const updated = await updateTeam(selectedTeam.id, {
          name: editName,
          description: editDescription,
          agent_ids: editAgentIds,
          lead_agent_id: editLeadAgentId || undefined,
        });
        setTeams((prev) => prev.map((t) => (t.id === updated.id ? updated : t)));
        selectTeam(updated);
      }
    } catch (err) {
      console.error("Failed to save team:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedTeam || !confirm("Delete this team?")) return;
    setIsLoading(true);

    try {
      await deleteTeam(selectedTeam.id);
      setTeams((prev) => prev.filter((t) => t.id !== selectedTeam.id));
      selectTeam(null);
    } catch (err) {
      console.error("Failed to delete team:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleAgent = (agentName: string) => {
    setEditAgentIds((prev) =>
      prev.includes(agentName)
        ? prev.filter((id) => id !== agentName)
        : [...prev, agentName]
    );
  };

  return (
    <div className={styles.container}>
      <div className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <span>Teams</span>
          <button className={styles.addBtn} onClick={startCreating}>
            +
          </button>
        </div>
        <div className={styles.teamList}>
          {teams.length === 0 ? (
            <div className={styles.empty}>No teams yet</div>
          ) : (
            teams.map((team) => (
              <div
                key={team.id}
                className={`${styles.teamItem} ${selectedTeam?.id === team.id ? styles.selected : ""}`}
                onClick={() => selectTeam(team)}
              >
                <div className={styles.teamName}>{team.name}</div>
                <div className={styles.teamMeta}>
                  {team.agent_ids?.length || 0} agents
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className={styles.editor}>
        {!selectedTeam && !isCreating ? (
          <div className={styles.placeholder}>
            <p>Select a team to edit or create a new one</p>
          </div>
        ) : (
          <div className={styles.form}>
            <div className={styles.formGroup}>
              <label>Team Name</label>
              <input
                type="text"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="e.g., Marketing Team"
                className={styles.input}
              />
            </div>

            <div className={styles.formGroup}>
              <label>Description</label>
              <textarea
                value={editDescription}
                onChange={(e) => setEditDescription(e.target.value)}
                placeholder="Describe what this team does..."
                className={styles.textarea}
                rows={3}
              />
            </div>

            <div className={styles.formGroup}>
              <label>Team Members</label>
              <div className={styles.agentGrid}>
                {agents.map((agent) => {
                  const isSelected = editAgentIds.includes(agent.id);
                  const isLead = editLeadAgentId === agent.id;
                  return (
                    <div
                      key={agent.id}
                      className={`${styles.agentCard} ${isSelected ? styles.agentSelected : ""}`}
                      onClick={() => toggleAgent(agent.id)}
                    >
                      <div className={styles.agentInfo}>
                        <div className={styles.agentName}>{agent.name}</div>
                        <div className={styles.agentDesc}>{agent.description}</div>
                      </div>
                      {isSelected && (
                        <button
                          className={`${styles.leadBtn} ${isLead ? styles.leadActive : ""}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditLeadAgentId(isLead ? null : agent.id);
                          }}
                          title={isLead ? "Remove as lead" : "Set as lead"}
                        >
                          {isLead ? "Lead" : "Set Lead"}
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className={styles.actions}>
              <button
                className={styles.saveBtn}
                onClick={handleSave}
                disabled={isLoading || !editName.trim()}
              >
                {isLoading ? "Saving..." : isCreating ? "Create Team" : "Save Changes"}
              </button>
              {selectedTeam && (
                <button
                  className={styles.deleteBtn}
                  onClick={handleDelete}
                  disabled={isLoading}
                >
                  Delete
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
