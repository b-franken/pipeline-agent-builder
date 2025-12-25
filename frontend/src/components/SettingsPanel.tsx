"use client";

import { useState, useEffect, useCallback } from "react";
import {
  fetchSettings,
  updateSettings,
  fetchModelsForProvider,
  type SystemSettings,
} from "@/lib/api";
import styles from "./SettingsPanel.module.css";

export default function SettingsPanel() {
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [models, setModels] = useState<Array<{ id: string; name: string }>>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Form state
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [openaiKey, setOpenaiKey] = useState("");
  const [anthropicKey, setAnthropicKey] = useState("");
  const [googleKey, setGoogleKey] = useState("");
  const [ollamaHost, setOllamaHost] = useState("");

  const loadSettings = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await fetchSettings();
      setSettings(data);
      setProvider(data.provider);
      setModel(data.model);
      setModels(data.available_models);
      setOllamaHost(data.ollama_host);
    } catch (err) {
      setError("Failed to load settings");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  // Load models when provider changes
  useEffect(() => {
    if (provider) {
      fetchModelsForProvider(provider)
        .then((data) => {
          setModels(data);
          // Reset model if not in new list
          if (!data.some((m) => m.id === model)) {
            setModel(data[0]?.id || "");
          }
        })
        .catch(() => setModels([]));
    }
  }, [provider]);

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const updates: Record<string, string> = {};

      if (provider !== settings?.provider) updates.provider = provider;
      if (model !== settings?.model) updates.model = model;
      if (openaiKey) updates.openai_api_key = openaiKey;
      if (anthropicKey) updates.anthropic_api_key = anthropicKey;
      if (googleKey) updates.google_api_key = googleKey;
      if (ollamaHost !== settings?.ollama_host) updates.ollama_host = ollamaHost;

      if (Object.keys(updates).length === 0) {
        setSuccess("No changes to save");
        setIsSaving(false);
        return;
      }

      const result = await updateSettings(updates);
      setSuccess(result.message);

      // Clear key inputs after save
      setOpenaiKey("");
      setAnthropicKey("");
      setGoogleKey("");

      // Reload settings
      await loadSettings();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save settings");
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <div className={styles.panel}>
        <div className={styles.loading}>Loading settings...</div>
      </div>
    );
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span>Settings</span>
      </div>

      {error && <div className={styles.error}>{error}</div>}
      {success && <div className={styles.success}>{success}</div>}

      <div className={styles.section}>
        <h3>LLM Provider</h3>
        <div className={styles.field}>
          <label>Provider</label>
          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            className={styles.select}
          >
            {settings?.available_providers.map((p) => (
              <option key={p} value={p}>
                {p.charAt(0).toUpperCase() + p.slice(1)}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.field}>
          <label>Model</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className={styles.select}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </div>

        {provider === "ollama" && (
          <div className={styles.field}>
            <label>Ollama Host</label>
            <input
              type="text"
              value={ollamaHost}
              onChange={(e) => setOllamaHost(e.target.value)}
              placeholder="http://localhost:11434"
              className={styles.input}
            />
          </div>
        )}
      </div>

      <div className={styles.section}>
        <h3>API Keys</h3>
        <p className={styles.hint}>
          Enter API keys to enable providers. Keys are stored securely.
        </p>

        <div className={styles.field}>
          <label>
            OpenAI API Key
            {settings?.has_openai_key && (
              <span className={styles.configured}> (configured)</span>
            )}
          </label>
          <input
            type="password"
            value={openaiKey}
            onChange={(e) => setOpenaiKey(e.target.value)}
            placeholder={settings?.has_openai_key ? "********" : "sk-..."}
            className={styles.input}
          />
        </div>

        <div className={styles.field}>
          <label>
            Anthropic API Key
            {settings?.has_anthropic_key && (
              <span className={styles.configured}> (configured)</span>
            )}
          </label>
          <input
            type="password"
            value={anthropicKey}
            onChange={(e) => setAnthropicKey(e.target.value)}
            placeholder={settings?.has_anthropic_key ? "********" : "sk-ant-..."}
            className={styles.input}
          />
        </div>

        <div className={styles.field}>
          <label>
            Google API Key
            {settings?.has_google_key && (
              <span className={styles.configured}> (configured)</span>
            )}
          </label>
          <input
            type="password"
            value={googleKey}
            onChange={(e) => setGoogleKey(e.target.value)}
            placeholder={settings?.has_google_key ? "********" : "AIza..."}
            className={styles.input}
          />
        </div>
      </div>

      <div className={styles.actions}>
        <button
          onClick={handleSave}
          disabled={isSaving}
          className={styles.saveBtn}
        >
          {isSaving ? "Saving..." : "Save Settings"}
        </button>
      </div>
    </div>
  );
}
