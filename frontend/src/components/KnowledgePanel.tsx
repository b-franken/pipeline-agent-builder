"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  fetchDocuments,
  uploadDocument,
  uploadFolder,
  deleteDocument,
  clearData,
  type DocumentInfo,
} from "@/lib/api";
import styles from "./KnowledgePanel.module.css";

const SUPPORTED_TYPES = [
  ".pdf",
  ".docx",
  ".xlsx",
  ".pptx",
  ".md",
  ".txt",
  ".csv",
  ".json",
  ".html",
  ".py",
  ".js",
  ".ts",
  ".tsx",
  ".jsx",
  ".go",
  ".rs",
  ".java",
  ".c",
  ".cpp",
  ".h",
  ".css",
  ".scss",
  ".yaml",
  ".yml",
  ".toml",
  ".xml",
  ".sql",
  ".sh",
  ".bash",
  ".rb",
  ".php",
];

function getFileIcon(filename: string, contentType: string): string {
  const ext = filename.split(".").pop()?.toLowerCase() || "";

  // Code files
  if (["py"].includes(ext)) return "PY";
  if (["js", "jsx"].includes(ext)) return "JS";
  if (["ts", "tsx"].includes(ext)) return "TS";
  if (["go"].includes(ext)) return "GO";
  if (["rs"].includes(ext)) return "RS";
  if (["java"].includes(ext)) return "JAVA";
  if (["c", "cpp", "h"].includes(ext)) return "C";
  if (["rb"].includes(ext)) return "RB";
  if (["php"].includes(ext)) return "PHP";
  if (["css", "scss"].includes(ext)) return "CSS";
  if (["sql"].includes(ext)) return "SQL";
  if (["sh", "bash"].includes(ext)) return "SH";

  // Config files
  if (["yaml", "yml"].includes(ext)) return "YML";
  if (["toml"].includes(ext)) return "TOML";
  if (["xml"].includes(ext)) return "XML";

  // Documents
  if (contentType.includes("pdf") || ext === "pdf") return "PDF";
  if (contentType.includes("word") || ext === "docx") return "DOC";
  if (contentType.includes("sheet") || ext === "xlsx") return "XLS";
  if (contentType.includes("presentation") || ext === "pptx") return "PPT";
  if (ext === "md" || contentType.includes("markdown")) return "MD";
  if (ext === "csv" || contentType.includes("csv")) return "CSV";
  if (ext === "json" || contentType.includes("json")) return "JSON";
  if (ext === "html" || contentType.includes("html")) return "HTML";

  return "TXT";
}

function getFolderFromPath(path: string): string | null {
  const parts = path.split("/");
  if (parts.length > 1) {
    return parts[0];
  }
  return null;
}

interface GroupedDocs {
  folders: Map<string, DocumentInfo[]>;
  files: DocumentInfo[];
}

function groupDocuments(docs: DocumentInfo[]): GroupedDocs {
  const folders = new Map<string, DocumentInfo[]>();
  const files: DocumentInfo[] = [];

  for (const doc of docs) {
    const folder = getFolderFromPath(doc.filename);
    if (folder) {
      const existing = folders.get(folder) || [];
      existing.push(doc);
      folders.set(folder, existing);
    } else {
      files.push(doc);
    }
  }

  return { folders, files };
}

export default function KnowledgePanel() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const folderInputRef = useRef<HTMLInputElement>(null);

  const loadDocuments = useCallback(async () => {
    try {
      const docs = await fetchDocuments();
      setDocuments(docs);
    } catch (err) {
      console.error("Failed to load documents:", err);
    }
  }, []);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  const handleUpload = async (files: FileList | File[], basePath: string = "") => {
    setIsUploading(true);
    setError(null);
    setSuccess(null);

    const fileArray = Array.from(files);
    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < fileArray.length; i++) {
      const file = fileArray[i];
      setUploadProgress(`Uploading ${i + 1}/${fileArray.length}: ${file.name}`);

      try {
        // Get relative path if available (for folder uploads)
        const relativePath = (file as any).webkitRelativePath || "";
        const finalPath = relativePath || (basePath ? `${basePath}/${file.name}` : file.name);

        await uploadDocument(file, finalPath);
        successCount++;
      } catch (err) {
        console.error(`Failed to upload ${file.name}:`, err);
        errorCount++;
      }
    }

    setIsUploading(false);
    setUploadProgress("");
    await loadDocuments();

    if (successCount > 0) {
      setSuccess(`Uploaded ${successCount} file(s)`);
    }
    if (errorCount > 0) {
      setError(`Failed to upload ${errorCount} file(s)`);
    }
  };

  const handleFolderUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      // Filter to only supported file types
      const files = Array.from(e.target.files).filter(file => {
        const ext = "." + file.name.split(".").pop()?.toLowerCase();
        return SUPPORTED_TYPES.includes(ext);
      });

      if (files.length === 0) {
        setError("No supported files found in folder");
        return;
      }

      await handleUpload(files);
    }
  };

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const items = e.dataTransfer.items;
      if (!items) {
        if (e.dataTransfer.files.length > 0) {
          handleUpload(e.dataTransfer.files);
        }
        return;
      }

      // Handle folder drops
      const files: File[] = [];

      const processEntry = async (entry: FileSystemEntry, path: string = ""): Promise<void> => {
        if (entry.isFile) {
          const fileEntry = entry as FileSystemFileEntry;
          return new Promise((resolve) => {
            fileEntry.file((file) => {
              const ext = "." + file.name.split(".").pop()?.toLowerCase();
              if (SUPPORTED_TYPES.includes(ext)) {
                // Add path info to file
                Object.defineProperty(file, 'webkitRelativePath', {
                  value: path ? `${path}/${file.name}` : file.name,
                  writable: false
                });
                files.push(file);
              }
              resolve();
            });
          });
        } else if (entry.isDirectory) {
          const dirEntry = entry as FileSystemDirectoryEntry;
          const dirReader = dirEntry.createReader();

          return new Promise((resolve) => {
            dirReader.readEntries(async (entries) => {
              const newPath = path ? `${path}/${entry.name}` : entry.name;
              await Promise.all(entries.map(e => processEntry(e, newPath)));
              resolve();
            });
          });
        }
      };

      // Process all dropped items
      const entries: FileSystemEntry[] = [];
      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry();
        if (entry) entries.push(entry);
      }

      if (entries.length > 0) {
        setIsUploading(true);
        setUploadProgress("Scanning folder...");

        await Promise.all(entries.map(e => processEntry(e)));

        if (files.length > 0) {
          await handleUpload(files);
        } else {
          setIsUploading(false);
          setUploadProgress("");
          setError("No supported files found");
        }
      }
    },
    []
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleUpload(e.target.files);
    }
  };

  const handleDelete = async (docId: string, filename: string) => {
    if (!confirm(`Delete "${filename}"?`)) return;

    try {
      await deleteDocument(docId);
      await loadDocuments();
      setSuccess(`Deleted ${filename}`);
    } catch (err) {
      setError(`Failed to delete ${filename}`);
    }
  };

  const handleClearAll = async () => {
    if (!confirm("Clear all documents from the knowledge base?")) return;

    try {
      const result = await clearData("knowledge");
      await loadDocuments();
      setSuccess(result.message);
    } catch (err) {
      setError("Failed to clear knowledge base");
    }
  };

  const toggleFolder = (folder: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(folder)) {
        next.delete(folder);
      } else {
        next.add(folder);
      }
      return next;
    });
  };

  const grouped = groupDocuments(documents);
  const totalChunks = documents.reduce((acc, doc) => acc + doc.chunk_count, 0);

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span>Knowledge Base</span>
        <div className={styles.actions}>
          <span className={styles.count}>{documents.length} docs · {totalChunks} chunks</span>
          {documents.length > 0 && (
            <button className={styles.clearBtn} onClick={handleClearAll}>
              Clear All
            </button>
          )}
        </div>
      </div>

      {/* Upload Zone */}
      <div
        className={`${styles.dropzone} ${isDragging ? styles.dragging : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          type="file"
          id="file-upload"
          className={styles.fileInput}
          multiple
          accept={SUPPORTED_TYPES.join(",")}
          onChange={handleFileSelect}
          disabled={isUploading}
        />
        <input
          type="file"
          ref={folderInputRef}
          className={styles.fileInput}
          {...{ webkitdirectory: "", directory: "" } as any}
          onChange={handleFolderUpload}
          disabled={isUploading}
        />
        <label htmlFor="file-upload" className={styles.dropLabel}>
          {isUploading ? (
            <span className={styles.uploading}>{uploadProgress || "Uploading..."}</span>
          ) : (
            <>
              <span className={styles.icon}>+</span>
              <span>Drop files or folders here</span>
              <span className={styles.hint}>
                Or click to upload files
              </span>
            </>
          )}
        </label>
      </div>

      {/* Folder Upload Button */}
      <button
        className={styles.folderBtn}
        onClick={() => folderInputRef.current?.click()}
        disabled={isUploading}
      >
        Upload Folder
      </button>

      {/* Messages */}
      {error && <div className={styles.error}>{error}</div>}
      {success && <div className={styles.success}>{success}</div>}

      {/* Document List */}
      <div className={styles.documents}>
        {documents.length === 0 ? (
          <div className={styles.empty}>
            No documents uploaded yet. Upload files or folders to enable RAG.
          </div>
        ) : (
          <>
            {/* Folders */}
            {Array.from(grouped.folders.entries()).map(([folder, docs]) => (
              <div key={folder} className={styles.folder}>
                <div
                  className={styles.folderHeader}
                  onClick={() => toggleFolder(folder)}
                >
                  <span className={styles.folderIcon}>
                    {expandedFolders.has(folder) ? "📂" : "📁"}
                  </span>
                  <span className={styles.folderName}>{folder}</span>
                  <span className={styles.folderCount}>{docs.length} files</span>
                </div>
                {expandedFolders.has(folder) && (
                  <div className={styles.folderContents}>
                    {docs.map((doc) => (
                      <div key={doc.id} className={styles.document}>
                        <div className={styles.docIcon}>
                          {getFileIcon(doc.filename, doc.content_type)}
                        </div>
                        <div className={styles.docInfo}>
                          <span className={styles.docName}>
                            {doc.filename.split("/").pop()}
                          </span>
                          <span className={styles.docMeta}>{doc.chunk_count} chunks</span>
                        </div>
                        <button
                          className={styles.deleteBtn}
                          onClick={() => handleDelete(doc.id, doc.filename)}
                          title="Delete document"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {/* Individual Files */}
            {grouped.files.map((doc) => (
              <div key={doc.id} className={styles.document}>
                <div className={styles.docIcon}>
                  {getFileIcon(doc.filename, doc.content_type)}
                </div>
                <div className={styles.docInfo}>
                  <span className={styles.docName}>{doc.filename}</span>
                  <span className={styles.docMeta}>{doc.chunk_count} chunks</span>
                </div>
                <button
                  className={styles.deleteBtn}
                  onClick={() => handleDelete(doc.id, doc.filename)}
                  title="Delete document"
                >
                  ×
                </button>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
