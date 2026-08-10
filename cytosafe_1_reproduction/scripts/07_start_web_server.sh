#!/bin/bash
# start_web_server.sh — Start the CytoSafe web server
#
# Usage:
#   chmod +x start_web_server.sh
#   ./start_web_server.sh
#
# Then open http://localhost:5050 in your browser.

set -e

PYTHON="/opt/homebrew/anaconda3/envs/cytosafe/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$SCRIPT_DIR/webapp/app.py"

echo "========================================"
echo "  CytoSafe Web Server"
echo "========================================"
echo "  Models dir : $SCRIPT_DIR/model"
echo "  URL        : http://localhost:5050"
echo "========================================"
echo ""

# Verify models exist
for CELL in 3T3 HEK293; do
    MODEL="$SCRIPT_DIR/../model/${CELL}_lgbm_ecfp4.joblib"
    if [ ! -f "$MODEL" ]; then
        echo "ERROR: Model not found: $MODEL"
        echo "Run 04_train_lgbm.py first."
        exit 1
    fi
done

echo "Models found. Starting server..."
echo "Press Ctrl+C to stop."
echo ""

cd "$SCRIPT_DIR/07_webapp"
"$PYTHON" app.py
