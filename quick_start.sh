#!/bin/bash

echo "🚀 REAL CRYPTO TRADING SYSTEM - QUICK START"
echo "============================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "⬆️  Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Quick start options:"
echo "  1. Paper trading:  python real_trading_launcher.py"
echo "  2. Market test:    python real_trading_system.py --test"
echo "  3. Check status:   python -c 'from real_trading_launcher import check_system_status; check_system_status()'"
echo ""
echo "💡 Always start with paper trading to learn the system!"
