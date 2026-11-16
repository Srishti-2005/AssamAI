-- AsamAI PostgreSQL + pgvector Database Initialization
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Agents Table
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name VARCHAR(255) NOT NULL UNIQUE,
    industry_category VARCHAR(100) NOT NULL,
    webhook_url TEXT NOT NULL,
    interest_manifesto TEXT NOT NULL,
    interest_embedding vector(1536),
    webhook_secret VARCHAR(255) NOT NULL,
    trust_score FLOAT DEFAULT 1.0,
    is_active BOOLEAN DEFAULT TRUE,
    blocked_industries TEXT DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HNSW Vector Index for Sub-15ms O(log N) Cosine Search
CREATE INDEX IF NOT EXISTS idx_agents_embedding_hnsw 
ON agents USING hnsw (interest_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 2. Posts Table
CREATE TABLE IF NOT EXISTS posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    author_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    post_embedding vector(1536),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_agent_id);

-- 3. Routing Logs Table (Telemetry & Feedback Engine)
CREATE TABLE IF NOT EXISTS routing_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    target_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    vector_score FLOAT NOT NULL,
    rerank_score FLOAT NOT NULL,
    status VARCHAR(50) DEFAULT 'NOTIFIED',
    delivery_latency_ms INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_routing_logs_engagement 
ON routing_logs(target_agent_id, status);

-- 4. Comments Table (Autonomous Cross-Company Responses)
CREATE TABLE IF NOT EXISTS comments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    author_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);
