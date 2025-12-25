-- Initialize PostgreSQL database for Agent Template
-- Enables pgvector extension and creates necessary tables

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for agent data
CREATE SCHEMA IF NOT EXISTS agent;

-- Grant permissions
GRANT ALL ON SCHEMA agent TO agent;
GRANT ALL ON ALL TABLES IN SCHEMA agent TO agent;
ALTER DEFAULT PRIVILEGES IN SCHEMA agent GRANT ALL ON TABLES TO agent;

-- Note: LangGraph checkpoint tables are created automatically by PostgresSaver.setup()
-- This script only ensures the database and extensions are ready
