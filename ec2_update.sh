#!/usr/bin/env bash
# ==============================================================================
# RouteIQ: Automated AWS EC2 Code Update & Rebuild Script
# VORTEXA 3.0 — Domain 2: AI Systems & Infrastructure (Problem Statement A-01)
# ==============================================================================
# Automates:
#   1. Fetching and pulling all latest commits from GitHub
#   2. Updating Python dependencies in the virtual environment
#   3. Re-verifying/training baseline estimator if updated
#   4. Gracefully restarting the RouteIQ systemd daemon
#   5. Reloading Nginx configuration if modified
#   6. Running end-to-end connectivity verification
# ==============================================================================

set -e

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "=================================================================="
echo "⚡ RouteIQ: Automated EC2 Update & Service Restart"
echo "   VORTEXA 3.0 — Fetch Latest GitHub Commits & Deploy"
echo "=================================================================="
echo -e "${NC}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_GROUP="$(id -gn "$ACTUAL_USER" 2>/dev/null || echo "$ACTUAL_USER")"

cd "$PROJECT_DIR"

# 1. Fetch Latest Commits from Git
echo -e "${BOLD}[1/4] Pulling Latest Commits from GitHub...${NC}"
if [ -d ".git" ]; then
    CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'main')"
    echo "Current branch: $CURRENT_BRANCH"
    git fetch origin
    git pull origin "$CURRENT_BRANCH"
    LATEST_COMMIT="$(git log -1 --oneline)"
    echo -e "${GREEN}✓ Updated to Commit:${NC} $LATEST_COMMIT"
else
    echo -e "${YELLOW}⚠️ Not a git repository. Skipping git pull.${NC}"
fi

# 2. Update Python Dependencies
echo -e "\n${BOLD}[2/4] Updating Dependencies in Virtual Environment...${NC}"
if [ -f "$VENV_DIR/bin/pip" ]; then
    export CMAKE_BUILD_PARALLEL_LEVEL=1
    "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
    "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/backend/requirements.txt"
    echo -e "${GREEN}✓ Dependencies up to date.${NC}"
else
    echo -e "${RED}❌ Error: Virtual environment not found at $VENV_DIR. Run ./ec2_setup.sh first.${NC}"
    exit 1
fi

# Re-calibrate estimator if needed
if [ ! -f "$PROJECT_DIR/backend/data/estimator_tfidf.joblib" ]; then
    echo "Calibrating baseline estimator..."
    "$VENV_DIR/bin/python" "$PROJECT_DIR/backend/scripts/train_estimator.py" || true
fi

# Fix ownership
if [ "$EUID" -eq 0 ]; then
    chown -R "$ACTUAL_USER:$ACTUAL_GROUP" "$PROJECT_DIR"
fi

# 3. Restart RouteIQ Systemd Daemon Service
echo -e "\n${BOLD}[3/4] Restarting RouteIQ Systemd Service...${NC}"
if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl restart routeiq
    sleep 2
    if sudo systemctl is-active --quiet routeiq; then
        echo -e "${GREEN}✓ routeiq.service restarted successfully and is active!${NC}"
    else
        echo -e "${RED}❌ routeiq.service failed to restart. Recent logs:${NC}"
        sudo journalctl -u routeiq -n 25 --no-pager
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️ systemctl not found. Please restart the process manually.${NC}"
fi

# Reload Nginx if present
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet nginx; then
    sudo systemctl reload nginx || sudo systemctl restart nginx
fi

# 4. End-to-End Health Check Verification
echo -e "\n${BOLD}[4/4] Verifying Gateway Health Check...${NC}"
HEALTH_CHECK=""
for i in {1..8}; do
    HEALTH_CHECK="$(curl -s -m 2 http://127.0.0.1:8000/health || curl -s -m 2 http://127.0.0.1/health || true)"
    if echo "$HEALTH_CHECK" | grep -q '"status":"ok"'; then
        break
    fi
    sleep 1
done

if echo "$HEALTH_CHECK" | grep -q '"status":"ok"'; then
    echo -e "${GREEN}✓ RouteIQ Gateway is ONLINE & HEALTHY: $HEALTH_CHECK${NC}"
else
    echo -e "${YELLOW}⚠️ Gateway response: $HEALTH_CHECK${NC}"
fi

PUBLIC_IP="$(curl -s -m 2 https://checkip.amazonaws.com 2>/dev/null || curl -s -m 2 https://ifconfig.me 2>/dev/null || echo 'YOUR_EC2_PUBLIC_IP')"

echo -e "\n${GREEN}${BOLD}=================================================================="
echo "✓ UPDATE AND RESTART COMPLETE!"
echo "==================================================================${NC}"
echo -e "  🌐 Web Dashboard:    ${CYAN}http://$PUBLIC_IP/${NC}"
echo -e "  📡 REST API Health:  ${CYAN}http://$PUBLIC_IP/health${NC}"
echo -e "  • Check status:      ${YELLOW}sudo systemctl status routeiq${NC}"
echo -e "  • Live logs:         ${YELLOW}sudo journalctl -u routeiq -f${NC}"
echo "=================================================================="
