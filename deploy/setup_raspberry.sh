#!/bin/bash
# Setup TradeVector P2 su Raspberry Pi
# Testato su Raspberry Pi OS (64-bit), Python 3.11+
# Eseguire come: bash deploy/setup_raspberry.sh

set -e
PROJECT_DIR="/home/pietro/tradevector-next"
VENV="$PROJECT_DIR/.venv"

echo "=== TradeVector P2 — Setup Raspberry Pi ==="

# 1. Dipendenze sistema
echo "[1/6] Dipendenze sistema..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-venv git -qq

# 2. Clone / pull progetto
git config --global --add safe.directory "$PROJECT_DIR"
if [ ! -d "$PROJECT_DIR/.git" ]; then
    echo "[2/6] Clone repository..."
    git clone https://github.com/pietropassantini/tradevector-next.git "$PROJECT_DIR"
else
    echo "[2/6] Update repository..."
    cd "$PROJECT_DIR" && git pull
fi

# 3. Virtual environment
echo "[3/6] Virtual environment..."
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r "$PROJECT_DIR/requirements.txt" -q

# 4. File .env (Telegram)
ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "[4/6] Creazione .env — inserisci credenziali Telegram:"
    read -p "  TELEGRAM_BOT_TOKEN: " BOT_TOKEN
    read -p "  TELEGRAM_CHAT_ID:   " CHAT_ID
    cat > "$ENV_FILE" << EOF
TELEGRAM_BOT_TOKEN=$BOT_TOKEN
TELEGRAM_CHAT_ID=$CHAT_ID
EOF
    chmod 600 "$ENV_FILE"
    echo "  .env salvato."
else
    echo "[4/6] .env già presente, skip."
fi

# 5. Directory logs
mkdir -p "$PROJECT_DIR/logs"

# 6. Systemd service + timer
echo "[5/6] Installazione systemd service..."
sudo cp "$PROJECT_DIR/deploy/tradevector-p2.service" /etc/systemd/system/
sudo cp "$PROJECT_DIR/deploy/tradevector-p2.timer" /etc/systemd/system/
# Il collector e' indipendente dalle strategie: l'archivio deve continuare ad
# accumularsi anche se lo scheduler P2 viene fermato.
sudo cp "$PROJECT_DIR/deploy/tradevector-collector.service" /etc/systemd/system/
sudo cp "$PROJECT_DIR/deploy/tradevector-collector.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tradevector-p2.timer
sudo systemctl start tradevector-p2.timer
sudo systemctl enable tradevector-collector.timer
sudo systemctl start tradevector-collector.timer

echo "[6/6] Test dry-run..."
cd "$PROJECT_DIR"
"$VENV/bin/python" scripts/p2_signal_scheduler.py --symbol BTCUSDT --dry-run

echo ""
echo "=== Setup completato ==="
echo "Timer attivo: ogni ora al minuto :05"
echo ""
echo "Comandi utili:"
echo "  sudo systemctl status tradevector-p2.timer    # stato timer"
echo "  sudo systemctl status tradevector-p2.service  # ultimo run"
echo "  sudo journalctl -u tradevector-p2.service -f  # log live"
echo "  tail -f $PROJECT_DIR/logs/p2_scheduler.log    # log file"
echo ""
echo "Test manuale:"
echo "  cd $PROJECT_DIR"
echo "  .venv/bin/python scripts/p2_signal_scheduler.py --symbol BTCUSDT --dry-run"
