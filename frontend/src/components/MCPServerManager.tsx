"use client";

import { useState, useEffect, useCallback } from "react";
import {
  fetchMCPServers,
  fetchMCPStatus,
  createMCPServer,
  updateMCPServer,
  deleteMCPServer,
  testMCPServer,
  fetchMCPPopularServers,
  fetchMCPCategories,
  type MCPServer,
  type MCPStatus,
  type MCPTestResult,
  type MCPPopularServer,
  type MCPCategory,
} from "@/lib/api";
import styles from "./MCPServerManager.module.css";

type TransportType = "stdio" | "http" | "streamable_http" | "sse";
type TabType = "configured" | "browse";

interface EditingServer {
  id: string;
  name: string;
  transport: TransportType;
  description: string;
  command: string;
  args: string[];
  url: string;
  headers: Record<string, string>;
  env_vars: Record<string, string>;
  is_active: boolean;
  isNew?: boolean;
}

const ICON_MAP: Record<string, string> = {
  github: "🐙",
  folder: "📁",
  database: "🗄️",
  brain: "🧠",
  globe: "🌐",
  search: "🔍",
  "message-square": "💬",
  monitor: "🖥️",
  "hard-drive": "💾",
  "git-branch": "🔀",
  image: "🖼️",
};

export default function MCPServerManager() {
  const [activeTab, setActiveTab] = useState<TabType>("configured");
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [status, setStatus] = useState<MCPStatus | null>(null);
  const [selectedServer, setSelectedServer] = useState<string | null>(null);
  const [editingServer, setEditingServer] = useState<EditingServer | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<MCPTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Browse tab state
  const [popularServers, setPopularServers] = useState<MCPPopularServer[]>([]);
  const [categories, setCategories] = useState<MCPCategory[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [isBrowseLoading, setIsBrowseLoading] = useState(false);

  const loadData = useCallback(async () => {
    try {
      setIsLoading(true);
      const [serverData, statusData] = await Promise.all([
        fetchMCPServers(),
        fetchMCPStatus(),
      ]);
      setServers(serverData);
      setStatus(statusData);
    } catch (err) {
      setError("Failed to load MCP servers");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadBrowseData = useCallback(async () => {
    try {
      setIsBrowseLoading(true);
      const [popularData, categoryData] = await Promise.all([
        fetchMCPPopularServers(selectedCategory || undefined),
        fetchMCPCategories(),
      ]);
      setPopularServers(popularData);
      setCategories(categoryData);
    } catch (err) {
      console.error("Failed to load browse data:", err);
    } finally {
      setIsBrowseLoading(false);
    }
  }, [selectedCategory]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (activeTab === "browse") {
      loadBrowseData();
    }
  }, [activeTab, loadBrowseData]);

  const handleSelectServer = (serverId: string) => {
    const server = servers.find((s) => s.id === serverId);
    if (server) {
      setSelectedServer(serverId);
      setEditingServer({
        id: server.id,
        name: server.name,
        transport: server.transport,
        description: server.description,
        command: server.command || "",
        args: server.args || [],
        url: server.url || "",
        headers: server.headers || {},
        env_vars: {},
        is_active: server.is_active,
      });
      setError(null);
      setSuccess(null);
      setTestResult(null);
    }
  };

  const handleNewServer = () => {
    setSelectedServer(null);
    setEditingServer({
      id: "",
      name: "",
      transport: "stdio",
      description: "",
      command: "",
      args: [],
      url: "",
      headers: {},
      env_vars: {},
      is_active: true,
      isNew: true,
    });
    setError(null);
    setSuccess(null);
    setTestResult(null);
  };

  const handleInstallPopular = (server: MCPPopularServer) => {
    // Check if server with same name already exists
    if (servers.some((s) => s.name === server.id)) {
      setError(`Server "${server.name}" is already configured`);
      setActiveTab("configured");
      return;
    }

    // Pre-fill form with popular server config
    setActiveTab("configured");
    setSelectedServer(null);
    setEditingServer({
      id: "",
      name: server.id,
      transport: server.transport,
      description: server.description,
      command: server.command,
      args: [...server.args],
      url: "",
      headers: {},
      env_vars: { ...server.env_vars },
      is_active: true,
      isNew: true,
    });
    setError(null);
    setSuccess(null);
    setTestResult(null);
  };

  const handleSave = async () => {
    if (!editingServer) return;

    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      if (editingServer.isNew) {
        if (!editingServer.name) {
          throw new Error("Name is required");
        }
        if (editingServer.transport === "stdio" && !editingServer.command) {
          throw new Error("Command is required for stdio transport");
        }
        if (
          ["http", "streamable_http", "sse"].includes(editingServer.transport) &&
          !editingServer.url
        ) {
          throw new Error("URL is required for HTTP-based transports");
        }

        await createMCPServer({
          name: editingServer.name,
          transport: editingServer.transport,
          description: editingServer.description || undefined,
          command: editingServer.command || undefined,
          args: editingServer.args.length > 0 ? editingServer.args : undefined,
          url: editingServer.url || undefined,
          headers:
            Object.keys(editingServer.headers).length > 0
              ? editingServer.headers
              : undefined,
          env_vars:
            Object.keys(editingServer.env_vars).length > 0
              ? editingServer.env_vars
              : undefined,
        });
        setSuccess("Server created successfully");
      } else {
        await updateMCPServer(editingServer.id, {
          name: editingServer.name,
          transport: editingServer.transport,
          description: editingServer.description,
          command: editingServer.command || undefined,
          args: editingServer.args,
          url: editingServer.url || undefined,
          headers: editingServer.headers,
          env_vars: editingServer.env_vars,
          is_active: editingServer.is_active,
        });
        setSuccess("Server updated successfully");
      }

      await loadData();
      if (editingServer.isNew) {
        setEditingServer({ ...editingServer, isNew: false });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save server");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!editingServer || editingServer.isNew) return;

    if (!confirm(`Delete "${editingServer.name}"? This cannot be undone.`)) return;

    try {
      await deleteMCPServer(editingServer.id);
      setSuccess("Server deleted");
      setSelectedServer(null);
      setEditingServer(null);
      await loadData();
    } catch (err) {
      setError("Failed to delete server");
    }
  };

  const handleTest = async () => {
    if (!editingServer || editingServer.isNew) return;

    setIsTesting(true);
    setTestResult(null);
    setError(null);

    try {
      const result = await testMCPServer(editingServer.id);
      setTestResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Test failed");
    } finally {
      setIsTesting(false);
    }
  };

  const handleArgsChange = (value: string) => {
    if (!editingServer) return;
    const args = value
      .split("\n")
      .map((a) => a.trim())
      .filter((a) => a);
    setEditingServer({ ...editingServer, args });
  };

  const handleHeadersChange = (value: string) => {
    if (!editingServer) return;
    const headers: Record<string, string> = {};
    value.split("\n").forEach((line) => {
      const idx = line.indexOf(":");
      if (idx > 0) {
        const key = line.substring(0, idx).trim();
        const val = line.substring(idx + 1).trim();
        if (key) headers[key] = val;
      }
    });
    setEditingServer({ ...editingServer, headers });
  };

  const handleEnvVarsChange = (value: string) => {
    if (!editingServer) return;
    const env_vars: Record<string, string> = {};
    value.split("\n").forEach((line) => {
      const idx = line.indexOf("=");
      if (idx > 0) {
        const key = line.substring(0, idx).trim();
        const val = line.substring(idx + 1).trim();
        if (key) env_vars[key] = val;
      }
    });
    setEditingServer({ ...editingServer, env_vars });
  };

  const hasRequiredEnvVars = (server: MCPPopularServer) => {
    return Object.values(server.env_vars).some((v) => v === "");
  };

  if (isLoading) {
    return (
      <div className={styles.panel}>
        <div className={styles.placeholder}>Loading MCP servers...</div>
      </div>
    );
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span>MCP Server Configuration</span>
          {status && (
            <span className={styles.statusBadge}>
              {status.available ? (
                <>
                  {status.configured_servers.length} server(s), {status.total_tools} tool(s)
                </>
              ) : (
                "MCP not available"
              )}
            </span>
          )}
        </div>
        {activeTab === "configured" && (
          <button onClick={handleNewServer} className={styles.addBtn}>
            + New Server
          </button>
        )}
      </div>

      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${activeTab === "configured" ? styles.active : ""}`}
          onClick={() => setActiveTab("configured")}
        >
          Configured ({servers.length})
        </button>
        <button
          className={`${styles.tab} ${activeTab === "browse" ? styles.active : ""}`}
          onClick={() => setActiveTab("browse")}
        >
          Browse Popular
        </button>
      </div>

      {activeTab === "configured" ? (
        <div className={styles.layout}>
          <div className={styles.serverList}>
            {servers.length === 0 ? (
              <div className={styles.emptyList}>No MCP servers configured</div>
            ) : (
              servers.map((server) => (
                <div
                  key={server.id}
                  className={`${styles.serverItem} ${selectedServer === server.id ? styles.selected : ""} ${!server.is_active ? styles.inactive : ""}`}
                  onClick={() => handleSelectServer(server.id)}
                >
                  <div className={styles.serverName}>
                    {server.name}
                    <span className={styles.transport}>{server.transport}</span>
                  </div>
                  <div className={styles.serverDesc}>
                    {server.description || "No description"}
                  </div>
                </div>
              ))
            )}
          </div>

          <div className={styles.editor}>
            {editingServer ? (
              <>
                {error && <div className={styles.error}>{error}</div>}
                {success && <div className={styles.success}>{success}</div>}
                {testResult && (
                  <div
                    className={testResult.success ? styles.success : styles.error}
                  >
                    {testResult.success
                      ? `Connected: ${testResult.tool_count} tool(s) available`
                      : `Failed: ${testResult.error}`}
                  </div>
                )}

                <div className={styles.field}>
                  <label>Name</label>
                  <input
                    type="text"
                    value={editingServer.name}
                    onChange={(e) =>
                      setEditingServer({ ...editingServer, name: e.target.value })
                    }
                    placeholder="my-mcp-server"
                    className={styles.input}
                  />
                </div>

                <div className={styles.field}>
                  <label>Transport</label>
                  <select
                    value={editingServer.transport}
                    onChange={(e) =>
                      setEditingServer({
                        ...editingServer,
                        transport: e.target.value as TransportType,
                      })
                    }
                    className={styles.select}
                  >
                    <option value="stdio">stdio (Local process)</option>
                    <option value="http">http</option>
                    <option value="streamable_http">streamable_http</option>
                    <option value="sse">sse (Server-Sent Events)</option>
                  </select>
                </div>

                <div className={styles.field}>
                  <label>Description</label>
                  <input
                    type="text"
                    value={editingServer.description}
                    onChange={(e) =>
                      setEditingServer({
                        ...editingServer,
                        description: e.target.value,
                      })
                    }
                    placeholder="What this server does..."
                    className={styles.input}
                  />
                </div>

                {editingServer.transport === "stdio" && (
                  <>
                    <div className={styles.field}>
                      <label>Command</label>
                      <input
                        type="text"
                        value={editingServer.command}
                        onChange={(e) =>
                          setEditingServer({
                            ...editingServer,
                            command: e.target.value,
                          })
                        }
                        placeholder="npx, python, node, etc."
                        className={styles.input}
                      />
                    </div>

                    <div className={styles.field}>
                      <label>Arguments (one per line)</label>
                      <textarea
                        value={editingServer.args.join("\n")}
                        onChange={(e) => handleArgsChange(e.target.value)}
                        placeholder="-m&#10;mcp_server&#10;--option"
                        className={styles.textarea}
                        rows={3}
                      />
                    </div>

                    <div className={styles.field}>
                      <label>Environment Variables (KEY=value, one per line)</label>
                      <textarea
                        value={Object.entries(editingServer.env_vars)
                          .map(([k, v]) => `${k}=${v}`)
                          .join("\n")}
                        onChange={(e) => handleEnvVarsChange(e.target.value)}
                        placeholder="API_KEY=xxx&#10;DEBUG=true"
                        className={styles.textarea}
                        rows={2}
                      />
                    </div>
                  </>
                )}

                {["http", "streamable_http", "sse"].includes(
                  editingServer.transport
                ) && (
                  <>
                    <div className={styles.field}>
                      <label>URL</label>
                      <input
                        type="text"
                        value={editingServer.url}
                        onChange={(e) =>
                          setEditingServer({
                            ...editingServer,
                            url: e.target.value,
                          })
                        }
                        placeholder="http://localhost:8080/mcp"
                        className={styles.input}
                      />
                    </div>

                    <div className={styles.field}>
                      <label>Headers (Key: value, one per line)</label>
                      <textarea
                        value={Object.entries(editingServer.headers)
                          .map(([k, v]) => `${k}: ${v}`)
                          .join("\n")}
                        onChange={(e) => handleHeadersChange(e.target.value)}
                        placeholder="Authorization: Bearer xxx&#10;X-Custom: value"
                        className={styles.textarea}
                        rows={2}
                      />
                    </div>
                  </>
                )}

                {!editingServer.isNew && (
                  <div className={styles.field}>
                    <label className={styles.toggleLabel}>
                      <input
                        type="checkbox"
                        checked={editingServer.is_active}
                        onChange={(e) =>
                          setEditingServer({
                            ...editingServer,
                            is_active: e.target.checked,
                          })
                        }
                      />
                      <span>Server Active</span>
                    </label>
                  </div>
                )}

                <div className={styles.actions}>
                  <button
                    onClick={handleSave}
                    disabled={isSaving}
                    className={styles.saveBtn}
                  >
                    {isSaving
                      ? "Saving..."
                      : editingServer.isNew
                        ? "Create Server"
                        : "Save Changes"}
                  </button>
                  {!editingServer.isNew && (
                    <>
                      <button
                        onClick={handleTest}
                        disabled={isTesting}
                        className={styles.testBtn}
                      >
                        {isTesting ? "Testing..." : "Test Connection"}
                      </button>
                      <button onClick={handleDelete} className={styles.deleteBtn}>
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </>
            ) : (
              <div className={styles.placeholder}>
                Select a server to edit or create a new one
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className={styles.browseSection}>
          <div className={styles.categoryFilter}>
            <button
              className={`${styles.categoryBtn} ${selectedCategory === null ? styles.active : ""}`}
              onClick={() => setSelectedCategory(null)}
            >
              All
            </button>
            {categories.map((cat) => (
              <button
                key={cat.id}
                className={`${styles.categoryBtn} ${selectedCategory === cat.id ? styles.active : ""}`}
                onClick={() => setSelectedCategory(cat.id)}
              >
                {cat.name} ({cat.count})
              </button>
            ))}
          </div>

          {isBrowseLoading ? (
            <div className={styles.loadingSpinner}>Loading popular servers...</div>
          ) : (
            <div className={styles.serverGrid}>
              {popularServers.map((server) => (
                <div key={server.id} className={styles.popularCard}>
                  <div className={styles.popularCardHeader}>
                    <div className={styles.popularIcon}>
                      {ICON_MAP[server.icon] || "🔌"}
                    </div>
                    <div className={styles.popularInfo}>
                      <div className={styles.popularName}>{server.name}</div>
                      <div className={styles.popularPackage}>{server.package}</div>
                    </div>
                  </div>
                  <div className={styles.popularDescription}>
                    {server.description}
                  </div>
                  {hasRequiredEnvVars(server) && (
                    <div className={styles.envWarning}>
                      Requires API key or credentials
                    </div>
                  )}
                  <div className={styles.popularMeta}>
                    <span className={styles.popularCategory}>{server.category}</span>
                    <button
                      className={styles.installBtn}
                      onClick={() => handleInstallPopular(server)}
                    >
                      + Add Server
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
