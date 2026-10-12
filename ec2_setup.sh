#!/usr/bin/env bash
# ==============================================================================
# RouteIQ: Automated AWS EC2 Instance Setup & Deployment Script
# VORTEXA 3.0 — Domain 2: AI Systems & Infrastructure (Problem Statement A-01)
# ==============================================================================
# Automates:
#   1. System packages installation (Python 3, venv, pip, git, nginx, curl)
#   2. Virtual environment setup and Python dependency installation
#   3. Baseline estimator training & verification
#   4. Systemd daemon creation (/etc/systemd/system/routeiq.service) with RestartSec=5
#   5. Nginx reverse proxy configuration (port 80 -> port 8000) with SSE streaming
#   6. Automated service start, enablement & end-to-end health verification
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
echo "⚡ RouteIQ: Automated AWS EC2 Instance Setup & Deployment"
echo "   VORTEXA 3.0 — ModelRouter AI Inference Gateway"
echo "=================================================================="
echo -e "${NC}"

# 1. Determine execution paths & user
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_GROUP="$(id -gn "$ACTUAL_USER" 2>/dev/null || echo "$ACTUAL_USER")"

echo -e "${CYAN}📁 Project Directory:${NC} $PROJECT_DIR"
echo -e "${CYAN}👤 Running as User:${NC}   $ACTUAL_USER ($ACTUAL_GROUP)"

# 2. Check sudo privileges
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}⚠️ Notice: This script requires root privileges to install system packages, systemd unit, and Nginx.${NC}"
    echo "Re-running script with sudo..."
    exec sudo bash "$0" "$@"
fi

# 3. Detect Linux Distribution & Install Prerequisites
echo -e "\n${BOLD}[1/6] Installing System Prerequisites (Python 3, Nginx, Git, Curl)...${NC}"
if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y python3 python3-pip python3-venv python3-dev nginx git curl uvicorn build-essential cmake
elif command -v dnf >/dev/null 2>&1; then
    dnf update -y || true
    dnf install -y python3 python3-pip git nginx curl
elif command -v yum >/dev/null 2>&1; then
    yum update -y || true
    yum install -y python3 python3-pip git nginx curl
else
    echo -e "${RED}❌ Error: Unsupported package manager. Please install Python 3, venv, and Nginx manually.${NC}"
    exit 1
fi

# Ensure swap space exists (prevents C++ OOM during llama-cpp-python builds on low-RAM EC2 instances)
if [ ! -f /swapfile ] && [ $(free -m | awk '/^Mem:/{print $2}') -lt 8000 ]; then
    echo "Creating 4GB swapfile for compilation headroom..."
    fallocate -l 4G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=4096
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# 4. Pull Latest Code from Git
echo -e "\n${BOLD}[2/6] Checking Git Status & Latest Commits...${NC}"
cd "$PROJECT_DIR"
if [ -d ".git" ]; then
    echo -e "${GREEN}✓ Git repository detected. Fetching latest changes...${NC}"
    sudo -u "$ACTUAL_USER" git fetch origin || true
    CURRENT_BRANCH="$(sudo -u "$ACTUAL_USER" git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'main')"
    sudo -u "$ACTUAL_USER" git pull origin "$CURRENT_BRANCH" || true
    LATEST_COMMIT="$(sudo -u "$ACTUAL_USER" git log -1 --oneline 2>/dev/null || echo 'HEAD')"
    echo -e "${GREEN}✓ Current Commit:${NC} $LATEST_COMMIT"
fi

# 5. Set Up Python Virtual Environment & Install Dependencies
echo -e "\n${BOLD}[3/6] Setting Up Python Virtual Environment & Dependencies...${NC}"
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    sudo -u "$ACTUAL_USER" python3 -m venv "$VENV_DIR"
fi

# Install and upgrade pip packages
echo "Installing backend dependencies from backend/requirements.txt..."
export CMAKE_BUILD_PARALLEL_LEVEL=1
sudo -u "$ACTUAL_USER" CMAKE_BUILD_PARALLEL_LEVEL=1 "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
sudo -u "$ACTUAL_USER" CMAKE_BUILD_PARALLEL_LEVEL=1 "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/backend/requirements.txt"

# Ensure .env exists
if [ ! -f "$PROJECT_DIR/.env" ] && [ -f "$PROJECT_DIR/.env.example" ]; then
    echo "Creating .env from .env.example..."
    sudo -u "$ACTUAL_USER" cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
fi

# Calibrate/Train estimator if model file missing
if [ ! -f "$PROJECT_DIR/backend/data/estimator_tfidf.joblib" ]; then
    echo "Calibrating baseline estimator..."
    sudo -u "$ACTUAL_USER" "$VENV_DIR/bin/python" "$PROJECT_DIR/backend/scripts/train_estimator.py" || true
fi

# Fix ownership
chown -R "$ACTUAL_USER:$ACTUAL_GROUP" "$PROJECT_DIR"

# 6. Configure Systemd Daemon Service
echo -e "\n${BOLD}[4/6] Configuring Systemd Daemon Service (routeiq.service)...${NC}"
SERVICE_FILE="/etc/systemd/system/routeiq.service"

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=RouteIQ Intelligent Cost-, Latency- & Quality-Aware AI Inference Gateway
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$ACTUAL_USER
Group=$ACTUAL_GROUP
WorkingDirectory=$PROJECT_DIR
Environment="PYTHONPATH=$PROJECT_DIR"
Environment="PORT=8000"
Environment="HOST=0.0.0.0"
ExecStart=$VENV_DIR/bin/python -m backend.app.main --host 0.0.0.0 --port 8000

# Crash recovery: restart immediately after 5 seconds
Restart=always
RestartSec=5

# Logging & Limits
LimitNOFILE=65536
StandardOutput=journal
StandardError=journal
SyslogIdentifier=routeiq

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd daemon and enabling routeiq.service..."
systemctl daemon-reload
systemctl enable routeiq
systemctl restart routeiq

# 7. Configure Nginx Reverse Proxy (Port 80 -> Port 8000)
echo -e "\n${BOLD}[5/6] Configuring Nginx Reverse Proxy (Port 80 -> Port 8000)...${NC}"

NGINX_CONF_CONTENT='server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    client_max_body_size 50M;

    # Gzip Compression for fast UI loading
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;

        # WebSocket & SSE header forwarding
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable buffering for real-time Server-Sent Events (/events)
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
'

if [ -d "/etc/nginx/sites-available" ]; then
    echo "$NGINX_CONF_CONTENT" > /etc/nginx/sites-available/routeiq
    # Remove default site if present to prevent port 80 conflicts
    if [ -f "/etc/nginx/sites-enabled/default" ]; then
        rm -f /etc/nginx/sites-enabled/default
    fi
    ln -sf /etc/nginx/sites-available/routeiq /etc/nginx/sites-enabled/routeiq
elif [ -d "/etc/nginx/conf.d" ]; then
    echo "$NGINX_CONF_CONTENT" > /etc/nginx/conf.d/routeiq.conf
fi

echo "Testing Nginx configuration syntax..."
nginx -t

echo "Enabling and restarting Nginx..."
systemctl enable nginx
systemctl restart nginx

# 8. Automated End-to-End Health Verification
echo -e "\n${BOLD}[6/6] Verifying Deployment Health & Connectivity...${NC}"
sleep 2

# Check systemd service status
if systemctl is-active --quiet routeiq; then
    echo -e "${GREEN}✓ routeiq.service is ACTIVE and running!${NC}"
else
    echo -e "${RED}❌ routeiq.service failed to start. Journalctl logs:${NC}"
    journalctl -u routeiq -n 25 --no-pager
    exit 1
fi

# Query health check endpoint via localhost
HEALTH_CHECK=""
for i in {1..10}; do
    HEALTH_CHECK="$(curl -s -m 2 http://127.0.0.1:8000/health || curl -s -m 2 http://127.0.0.1/health || true)"
    if echo "$HEALTH_CHECK" | grep -q '"status":"ok"'; then
        break
    fi
    sleep 1
done

if echo "$HEALTH_CHECK" | grep -q '"status":"ok"'; then
    echo -e "${GREEN}✓ Backend Health Check PASSED: $HEALTH_CHECK${NC}"
else
    echo -e "${YELLOW}⚠️ Note: Health response: $HEALTH_CHECK${NC}"
fi

# Detect Public IP
PUBLIC_IP="$(curl -s -m 2 https://checkip.amazonaws.com 2>/dev/null || curl -s -m 2 https://ifconfig.me 2>/dev/null || echo 'YOUR_EC2_PUBLIC_IP')"

echo -e "\n${GREEN}${BOLD}=================================================================="
echo "🎉 RouteIQ DEPLOYMENT SUCCESSFUL!"
echo "==================================================================${NC}"
echo -e "${BOLD}Access your live instance:${NC}"
echo -e "  🌐 Web Dashboard:    ${CYAN}http://$PUBLIC_IP/${NC}"
echo -e "  📡 REST API Health:  ${CYAN}http://$PUBLIC_IP/health${NC}"
echo -e "  🚀 Route Inference:  ${CYAN}POST http://$PUBLIC_IP/route${NC}"
echo ""
echo -e "${BOLD}Useful Administration Commands:${NC}"
echo -e "  • Check status:      ${YELLOW}sudo systemctl status routeiq${NC}"
echo -e "  • View live logs:    ${YELLOW}sudo journalctl -u routeiq -f${NC}"
echo -e "  • Restart service:   ${YELLOW}sudo systemctl restart routeiq${NC}"
echo -e "  • Update code:       ${YELLOW}./ec2_update.sh${NC}"
echo "=================================================================="
