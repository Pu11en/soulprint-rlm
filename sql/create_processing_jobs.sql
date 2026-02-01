-- Create processing_jobs table for job recovery after server restarts
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, processing, complete, failed
    storage_path TEXT,  -- Path to raw JSON in user-imports bucket
    conversation_count INT,
    message_count INT,
    current_step TEXT,  -- downloading, chunking, embedding, synthesizing, generating, callback
    progress INT DEFAULT 0,  -- 0-100
    error_message TEXT,
    attempts INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Index for finding stuck jobs
CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_user_id ON processing_jobs(user_id);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_processing_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_processing_jobs_updated_at ON processing_jobs;
CREATE TRIGGER trigger_processing_jobs_updated_at
    BEFORE UPDATE ON processing_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_processing_jobs_updated_at();

-- RLS policies (service role bypasses these, but good to have)
ALTER TABLE processing_jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own jobs" ON processing_jobs
    FOR SELECT USING (auth.uid() = user_id);

-- Service role can do everything (used by RLM)
CREATE POLICY "Service role full access" ON processing_jobs
    FOR ALL USING (auth.role() = 'service_role');
