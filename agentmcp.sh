#!/bin/bash
# ==========================================
# AgentMCP — TurboQuant MCP Server Manager
# ==========================================
#   agentmcp.sh          → avvia il server MCP
#   agentmcp.sh --stop   → ferma il server MCP
# ==========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.agentmcp.pid"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8081}"
PYTHON_REQUIRED="3.11"
VENV_DIR="$SCRIPT_DIR/.venv"

# --- Colori ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --- Funzioni ---

check_python() {
    local PYTHON_CMD=""
    for py in python$PYTHON_REQUIRED python3.11 python3.10 python3; do
        if command -v "$py" &>/dev/null; then
            VERSION=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if [[ "$VERSION" == "3.10" || "$VERSION" == "3.11" ]]; then
                PYTHON_CMD="$py"
                break
            fi
        fi
    done
    if [ -z "$PYTHON_CMD" ]; then
        echo -e "${RED}❌ Python $PYTHON_REQUIRED o successivo non trovato.${NC}"
        echo "   Installare: sudo apt install python3.11"
        exit 1
    fi
    echo "$PYTHON_CMD"
}

setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}📦 Creo virtual env in $VENV_DIR ...${NC}"
        python3.11 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
}

install_deps() {
    if [ ! -f "$VENV_DIR/.deps_installed" ]; then
        pip install --quiet --upgrade pip
        pip install --quiet fastmcp uvicorn starlette httpx pydantic
        touch "$VENV_DIR/.deps_installed"
        echo "✅ Dipendenze installate"
    else
        python -c "import fastmcp; import uvicorn; import starlette" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "📦 Pacchetti mancanti, reinstallazione..."
            pip install --quiet fastmcp uvicorn starlette httpx pydantic
            touch "$VENV_DIR/.deps_installed"
        fi
    fi
}

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

start_server() {
    # Verifica se è già in esecuzione
    local existing_pid
    if existing_pid=$(is_running); then
        echo -e "${YELLOW}⚠️  Server già in esecuzione (PID: $existing_pid)${NC}"
        echo "   Usa $0 --stop per fermarlo."
        return 0
    fi

    echo "=== Verifica ambiente ==="
    local python_cmd
    python_cmd=$(check_python)
    echo -e "${GREEN}✅ Python: $python_cmd${NC}"

    setup_venv
    install_deps

    echo ""
    echo "🚀 Avvio AgentMCP server..."
    echo -e "   ${GREEN}URL: http://${SERVER_HOST}:${SERVER_PORT}/sse${NC}"
    echo ""

    # Avvia il server in background
    WORKSPACE_DIR="$SCRIPT_DIR" \
    SERVER_HOST="$SERVER_HOST" \
    SERVER_PORT="$SERVER_PORT" \
    nohup python server.py > "$SCRIPT_DIR/.agentmcp.log" 2>&1 &
    local server_pid=$!
    echo "$server_pid" > "$PID_FILE"

    # Attendere che il server sia pronto (max 10s)
    local retries=20
    while [ $retries -gt 0 ]; do
        if curl -s "http://${SERVER_HOST}:${SERVER_PORT}/sse" -X POST \
            -H "Content-Type: application/json" \
            -d '{"jsonrpc":"2.0","id":"ping","method":"ping","params":{}}' \
            >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Server AgentMCP avviato (PID: $server_pid)${NC}"
            return 0
        fi
        sleep 0.5
        retries=$((retries - 1))
    done

    # Se non è partito, controlla i log
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo -e "${RED}❌ Il server non è riuscito ad avviarsi.${NC}"
        echo "   Controlla i log: $SCRIPT_DIR/.agentmcp.log"
        rm -f "$PID_FILE"
        return 1
    fi

    echo -e "${YELLOW}⚠️  Server avviato ma non risponde ancora (PID: $server_pid)${NC}"
    echo "   Controlla i log: $SCRIPT_DIR/.agentmcp.log"
    return 0
}

stop_server() {
    if ! is_running &>/dev/null; then
        echo -e "${YELLOW}⚠️  Server AgentMCP non in esecuzione.${NC}"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid=$(cat "$PID_FILE")
    echo "🛑 Arresto server AgentMCP (PID: $pid)..."

    # TERM graceful
    kill -TERM "$pid" 2>/dev/null

    # Attende fino a 10 secondi
    local retries=20
    while [ $retries -gt 0 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 0.5
        retries=$((retries - 1))
    done

    # Force kill se necessario
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}Forzo l'arresto con SIGKILL...${NC}"
        kill -9 "$pid" 2>/dev/null
        sleep 0.5
    fi

    rm -f "$PID_FILE"
    echo -e "${GREEN}✅ Server AgentMCP fermato.${NC}"
}

show_status() {
    if is_running &>/dev/null; then
        local pid
        pid=$(cat "$PID_FILE")
        local url="http://${SERVER_HOST}:${SERVER_PORT}/sse"
        echo -e "${GREEN}✅ AgentMCP server${NC} — ${GREEN}IN ESECUZIONE${NC}"
        echo "   PID:   $pid"
        echo "   URL:   $url"
        echo "   Log:   $SCRIPT_DIR/.agentmcp.log"
    else
        echo -e "${YELLOW}⚠️  AgentMCP server${NC} — ${RED}FERMO${NC}"
        rm -f "$PID_FILE"
    fi
}

# --- MAIN ---

case "${1:-}" in
    --stop|-s)
        stop_server
        ;;
    --status|-st)
        show_status
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Avvia, ferma o mostra lo stato del server AgentMCP."
        echo ""
        echo "Options:"
        echo "  (no args)       Avvia il server MCP"
        echo "  --stop, -s      Ferma il server MCP"
        echo "  --status, -st   Mostra lo stato del server"
        echo "  --help, -h      Mostra questo aiuto"
        echo ""
        echo "Environment variables:"
        echo "  SERVER_HOST     Host di ascolto (default: 127.0.0.1)"
        echo "  SERVER_PORT     Porta di ascolto (default: 8081)"
        echo "  WORKSPACE_DIR   Directory sandbox (default: script dir)"
        ;;
    "")
        start_server
        ;;
    *)
        echo "❌ Opzione sconosciuta: $1"
        echo "Usa $0 --help per i comandi disponibili."
        exit 1
        ;;
esac
