"""Project configuration. Set your API keys here or use environment variables."""

PROVIDER = "openai"  # openai, anthropic, google, ollama

# API Keys (or set via environment variables)
OPENAI_API_KEY = ""
GOOGLE_API_KEY = ""
ANTHROPIC_API_KEY = ""

# Model configuration
MODEL = "gpt-4o"

# Ollama configuration (for local models)
OLLAMA_BASE_URL = "http://localhost:11434"
