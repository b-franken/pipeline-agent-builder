#!/bin/bash
# Ollama Model Initialization Script
# This script pulls the required models for local LLM operation

set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

echo "Waiting for Ollama to be ready..."
until curl -s "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
    echo "Ollama not ready yet, waiting..."
    sleep 2
done
echo "Ollama is ready!"

# Pull the main LLM model (default: llama3.2 - good balance of speed and quality)
LLM_MODEL="${OLLAMA_MODEL:-llama3.2}"
echo "Pulling LLM model: ${LLM_MODEL}..."
curl -X POST "${OLLAMA_HOST}/api/pull" -d "{\"name\": \"${LLM_MODEL}\"}"
echo "LLM model ${LLM_MODEL} ready!"

# Pull embedding model for local RAG (nomic-embed-text is excellent for RAG)
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
echo "Pulling embedding model: ${EMBED_MODEL}..."
curl -X POST "${OLLAMA_HOST}/api/pull" -d "{\"name\": \"${EMBED_MODEL}\"}"
echo "Embedding model ${EMBED_MODEL} ready!"

echo ""
echo "=== Ollama Local Setup Complete ==="
echo "LLM Model: ${LLM_MODEL}"
echo "Embedding Model: ${EMBED_MODEL}"
echo ""
echo "Available models:"
curl -s "${OLLAMA_HOST}/api/tags" | python3 -c "import sys, json; models = json.load(sys.stdin).get('models', []); [print(f'  - {m[\"name\"]}') for m in models]"
