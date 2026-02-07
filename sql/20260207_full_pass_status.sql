-- Migration: Add full pass tracking columns to user_profiles
-- Created: 2026-02-07
-- Phase: 04-pipeline-integration (Plan 04-02)
--
-- IMPORTANT: This migration MUST be executed in Supabase SQL Editor BEFORE deploying
-- the RLM service changes from Phase 4. The full pass pipeline writes to memory_md and
-- full_pass_status columns — if they don't exist, the pipeline will fail silently
-- (best-effort status updates).
--
-- NOTE: This migration may already be deployed (it exists in soulprint-landing from v1.2).
-- The IF NOT EXISTS pattern ensures idempotency — safe to re-run.
--
-- To execute:
-- 1. Open Supabase SQL Editor
-- 2. Paste this entire file
-- 3. Run the migration
-- 4. Verify columns exist: SELECT column_name FROM information_schema.columns WHERE table_name = 'user_profiles';

-- Add memory_md column for curated durable facts
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS memory_md TEXT;
COMMENT ON COLUMN public.user_profiles.memory_md IS 'MEMORY section - curated durable facts (preferences, projects, dates, beliefs, decisions)';

-- Add full_pass_status column with default 'pending'
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS full_pass_status TEXT DEFAULT 'pending';
COMMENT ON COLUMN public.user_profiles.full_pass_status IS 'Background full pass status: pending, processing, complete, failed';

-- Add timing columns for full pass tracking
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS full_pass_started_at TIMESTAMPTZ;
COMMENT ON COLUMN public.user_profiles.full_pass_started_at IS 'Timestamp when full pass background task started';

ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS full_pass_completed_at TIMESTAMPTZ;
COMMENT ON COLUMN public.user_profiles.full_pass_completed_at IS 'Timestamp when full pass background task completed (success or failure)';

-- Add error column for failure tracking
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS full_pass_error TEXT;
COMMENT ON COLUMN public.user_profiles.full_pass_error IS 'Error message if full pass failed (null if successful or in progress)';

-- Add check constraint for valid status values (IF NOT EXISTS not supported for constraints)
-- Use DO block to handle constraint existence check
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'full_pass_status_check'
    ) THEN
        ALTER TABLE public.user_profiles ADD CONSTRAINT full_pass_status_check
          CHECK (full_pass_status IN ('pending', 'processing', 'complete', 'failed'));
    END IF;
END $$;
