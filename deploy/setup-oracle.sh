#!/bin/bash
#
# Jack Foundation - Oracle Ampere A1 Setup Script
#
# This script sets up the Oracle Ampere A1 instance with:
# - llama.cpp for local LLM inference
# - Python environment for Jack agent
# - Jack Server (FastAPI)
# - Systemd services for auto-start
#
# Usage:
#   ./setup-oracle.sh
#
# Requirements:
#   - Oracle Ampere A1 instance (24GB RAM recommended)
#   - Ubuntu 22.04 or later
#   - Root/sudo access
#

set -e

echo "============================================"
echo "  Jack Foundation - Oracle Ampere Setup"
echo "============================================"

# Configuration
JACK_HOME="/opt/jack"
LLAMA_HOME="/opt/llama.cpp"
MODEL_DIR="/opt/models"
MODEL_NAME="deepseek-r1-distill-qwen-14b-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Please run as root or with sudo"
        exit 1
    fi
}

# Update system packages
update_system() {
    log_info "Updating system packages..."
    apt-get update && apt-get upgrade -y
}

# Install system dependencies
install_dependencies() {
    log_info "Installing system dependencies..."
    apt-get install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        python3 \
        python3-pip \
        python3-venv \
        nginx \
        certbot \
        python3-certbot-nginx \
        htop \
        iotop \
        unzip
}

# Build llama.cpp for ARM
build_llama_cpp() {
    log_info "Building llama.cpp for ARM..."

    if [ -d "$LLAMA_HOME" ]; then
        log_warn "llama.cpp directory exists, updating..."
        cd "$LLAMA_HOME"
        git pull
    else
        git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_HOME"
        cd "$LLAMA_HOME"
    fi

    # Build with optimizations for ARM
    mkdir -p build
    cd build
    cmake .. -DLLAMA_NATIVE=ON
    cmake --build . --config Release -j$(nproc)

    log_info "llama.cpp built successfully"
}

# Download the LLM model
download_model() {
    log_info "Downloading DeepSeek-R1-Distill-Qwen-14B model..."

    mkdir -p "$MODEL_DIR"

    if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
        log_warn "Model already exists, skipping download"
    else
        log_info "Downloading model (this may take a while)..."
        wget -c "$MODEL_URL" -O "$MODEL_DIR/$MODEL_NAME"
    fi

    log_info "Model downloaded to $MODEL_DIR/$MODEL_NAME"
}

# Setup Python environment
setup_python() {
    log_info "Setting up Python environment..."

    mkdir -p "$JACK_HOME"

    # Create virtual environment
    python3 -m venv "$JACK_HOME/venv"
    source "$JACK_HOME/venv/bin/activate"

    # Install requirements
    pip install --upgrade pip
    pip install \
        fastapi \
        uvicorn[standard] \
        pydantic \
        httpx \
        aiohttp \
        python-jose[cryptography] \
        passlib[bcrypt] \
        pyodbc \
        aioodbc

    log_info "Python environment ready"
}

# Create systemd service for llama.cpp server
create_llama_service() {
    log_info "Creating llama.cpp systemd service..."

    cat > /etc/systemd/system/llama-server.service << EOF
[Unit]
Description=llama.cpp OpenAI-compatible Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$LLAMA_HOME
ExecStart=$LLAMA_HOME/build/bin/llama-server \\
    -m $MODEL_DIR/$MODEL_NAME \\
    --host 127.0.0.1 \\
    --port 8080 \\
    -c 4096 \\
    -ngl 0 \\
    --threads $(nproc)
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable llama-server

    log_info "llama.cpp service created"
}

# Create systemd service for Jack Server
create_jack_service() {
    log_info "Creating Jack Server systemd service..."

    cat > /etc/systemd/system/jack-server.service << EOF
[Unit]
Description=Jack Foundation API Server
After=network.target llama-server.service
Wants=llama-server.service

[Service]
Type=simple
User=root
WorkingDirectory=$JACK_HOME
Environment="PATH=$JACK_HOME/venv/bin:/usr/bin"
Environment="LLM_PROVIDER=local"
Environment="LLM_BASE_URL=http://127.0.0.1:8080/v1"
Environment="LLM_MODEL=deepseek-r1-14b"
Environment="JWT_SECRET=CHANGE_ME_TO_SECURE_SECRET"
Environment="SERVER_PORT=8000"
ExecStart=$JACK_HOME/venv/bin/python -m uvicorn jack.server.app:create_app \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --factory
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable jack-server

    log_info "Jack Server service created"
}

# Configure nginx reverse proxy
setup_nginx() {
    log_info "Configuring nginx reverse proxy..."

    cat > /etc/nginx/sites-available/jack << EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Timeout for long-running requests
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
    }
}
EOF

    ln -sf /etc/nginx/sites-available/jack /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default

    nginx -t && systemctl restart nginx

    log_info "nginx configured"
}

# Configure firewall
setup_firewall() {
    log_info "Configuring firewall..."

    # Allow SSH, HTTP, HTTPS
    ufw allow 22/tcp
    ufw allow 80/tcp
    ufw allow 443/tcp
    ufw --force enable

    log_info "Firewall configured"
}

# Create environment file
create_env_file() {
    log_info "Creating environment configuration..."

    cat > "$JACK_HOME/.env" << EOF
# Jack Foundation Configuration
# IMPORTANT: Change JWT_SECRET to a secure value!

# LLM Provider (local, openai, anthropic)
LLM_PROVIDER=local
LLM_BASE_URL=http://127.0.0.1:8080/v1
LLM_MODEL=deepseek-r1-14b

# Authentication
JWT_SECRET=CHANGE_ME_TO_A_VERY_SECURE_SECRET_KEY_$(openssl rand -hex 16)
JWT_EXPIRY_HOURS=24

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG=false

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Database (optional - for MSSQL via TailScale)
# DB_SERVER=your-tailscale-ip
# DB_DATABASE=YourDatabase
# DB_USERNAME=your_username
# DB_PASSWORD=your_password
# DB_TRUSTED=false
EOF

    chmod 600 "$JACK_HOME/.env"

    log_info "Environment file created at $JACK_HOME/.env"
    log_warn "IMPORTANT: Edit $JACK_HOME/.env and set a secure JWT_SECRET!"
}

# Print summary
print_summary() {
    echo ""
    echo "============================================"
    echo "  Setup Complete!"
    echo "============================================"
    echo ""
    echo "Services installed:"
    echo "  - llama.cpp server (port 8080, local only)"
    echo "  - Jack Server (port 8000, via nginx on 80)"
    echo ""
    echo "Commands:"
    echo "  - Start llama: sudo systemctl start llama-server"
    echo "  - Start Jack:  sudo systemctl start jack-server"
    echo "  - View logs:   sudo journalctl -u jack-server -f"
    echo ""
    echo "Configuration:"
    echo "  - Environment: $JACK_HOME/.env"
    echo "  - Model path:  $MODEL_DIR/$MODEL_NAME"
    echo ""
    echo "IMPORTANT:"
    echo "  1. Edit $JACK_HOME/.env and set a secure JWT_SECRET"
    echo "  2. Copy your Jack Foundation code to $JACK_HOME"
    echo "  3. Start services: sudo systemctl start llama-server jack-server"
    echo ""
    echo "API will be available at: http://$(curl -s ifconfig.me)"
    echo "============================================"
}

# Main installation
main() {
    check_root
    update_system
    install_dependencies
    build_llama_cpp
    download_model
    setup_python
    create_llama_service
    create_jack_service
    setup_nginx
    setup_firewall
    create_env_file
    print_summary
}

main "$@"
