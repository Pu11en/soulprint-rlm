#!/bin/bash
# Verification script for embedding PATCH fix
# Run after Render deployment completes

RLM_URL="https://soulprint-landing.onrender.com"

echo "=== Testing RLM Embedding PATCH Fix ==="
echo ""

echo "1. Checking service health..."
curl -s "$RLM_URL/health" | jq '.'
echo ""

echo "2. Running isolated PATCH test..."
echo "   (INSERT -> EMBED -> PATCH -> VERIFY)"
PATCH_TEST=$(curl -s "$RLM_URL/test-patch")
echo "$PATCH_TEST" | jq '.'

# Check if test passed
SUCCESS=$(echo "$PATCH_TEST" | jq -r '.success')
if [ "$SUCCESS" = "true" ]; then
    echo "✓ Isolated PATCH test PASSED"
    echo "  - Embedding dimensions: $(echo "$PATCH_TEST" | jq -r '.embedding_dims')"
    echo "  - Saved dimensions: $(echo "$PATCH_TEST" | jq -r '.saved_dims')"
    echo ""
    echo "3. Ready to test full import flow with /process-full"
    echo "   Next: Send real ChatGPT export to verify at scale"
else
    echo "✗ Isolated PATCH test FAILED"
    echo "  - Step: $(echo "$PATCH_TEST" | jq -r '.step')"
    echo "  - Error: $(echo "$PATCH_TEST" | jq -r '.error')"
    echo ""
    echo "Fix did not resolve the issue. Need further investigation."
    exit 1
fi
