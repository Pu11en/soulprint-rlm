# Rollback Procedure

## When to Rollback

Rollback if ANY of these are true after deploying changes:

**Phase 7 (Prompt Foundation) - Deployed 2026-02-08:**
- AI responses are generic (ignoring personality traits, using banned phrases like "Great question!")
- /query endpoint returns 500 errors (prompt_helpers import failure)
- Render build fails with "ModuleNotFoundError: No module named 'prompt_helpers'"
- AI does not reference its name (uses "SoulPrint" instead of generated name)

**Phase 3 (Processor Infrastructure):**
- /health endpoint returns 503 (processor import failure)
- Existing endpoints (/process-full, /chat, /query) return 500 errors
- Render dashboard shows service as unhealthy
- Users report errors in import or chat flows

## How to Rollback

### Step 1: Find the last known good commit

```bash
git log --oneline -10
```

Look for the commit BEFORE the Phase 3 changes. It will be the last commit from Phase 2 (processor tests or Dockerfile update).

### Step 2: Revert to last good commit

```bash
# Revert the most recent commit (if only 1 Phase 3 commit)
git revert HEAD --no-edit
git push origin main

# OR revert multiple commits (if 2+ Phase 3 commits)
# Find the merge base (last Phase 2 commit SHA)
git revert HEAD~N..HEAD --no-edit  # Where N is number of Phase 3 commits
git push origin main
```

### Step 3: Verify on Render

1. Go to Render dashboard: https://dashboard.render.com
2. Watch the deploy triggered by git push
3. Verify /health returns 200 after deploy completes
4. Verify /process-full still works (test with curl)

```bash
# Quick health check
curl https://soulprint-landing.onrender.com/health

# Verify v1 pipeline still works
curl -X POST https://soulprint-landing.onrender.com/process-full \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "storage_path": "test"}'
```

### Step 4: Investigate

After rollback stabilizes production:
1. Check Render logs for error messages
2. Look for processor import errors in startup logs
3. Check if the issue is env vars, missing files, or code bugs
4. Fix in a new branch, test locally, then re-deploy

## What Gets Preserved During Rollback

- All database data (Supabase) is unaffected
- Processing jobs table tracks in-flight jobs for recovery
- v1 /process-full pipeline continues working
- No data loss — rollback only affects code deployment

## What Gets Lost During Rollback

**Phase 7 Rollback (Prompt Foundation):**
- Personalized AI prompts revert to hardcoded "You are SoulPrint" generic prompts
- Personality traits, communication style, and values are ignored in responses
- Anti-generic phrase blocking is removed (AI may use "Great question!", "I'd be happy to help!")
- Memory context referencing becomes less natural ("According to retrieved context..." instead of "Like we talked about...")
- AI stops using its generated name

**Phase 3 Rollback (Processor Infrastructure):**
- /process-full-v2 endpoint becomes unavailable
- Processor import validation in health check removed
- Lifespan startup validation removed (reverts to @app.on_event)
- Any in-flight v2 background tasks are killed on restart (tracked in processing_jobs for retry)
