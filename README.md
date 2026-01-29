# SoulPrint RLM Service

Memory-enhanced chat backend for SoulPrint.

## Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## Environment Variables

- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_KEY` - Supabase service role key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `ALERT_WEBHOOK` - (Optional) Webhook for failure alerts

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8100
```
# Trigger deploy Thu Jan 29 12:19:35 CST 2026
