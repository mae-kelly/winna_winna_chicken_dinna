#!/bin/bash

echo "🔧 SETTING UP TRADING SYSTEM ENVIRONMENT"
echo "========================================"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    touch .env
fi

# Backup existing .env
if [ -s .env ]; then
    echo "💾 Backing up existing .env to .env.backup"
    cp .env .env.backup
fi

# Set up environment variables
echo "🔑 Setting up environment variables..."

cat > .env << 'EOF'
# Trading System Configuration
TRADING_MODE=paper
LOG_LEVEL=INFO

# OKX API Configuration (Paper Trading - Safe Defaults)
OKX_API_KEY=paper_trading_key
OKX_SECRET_KEY=paper_trading_secret
OKX_PASSPHRASE=paper_trading_pass

# System Configuration
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false

# Trading Parameters
INITIAL_CAPITAL=1000
TARGET_MULTIPLIER=10
MAX_POSITIONS=3
POSITION_SIZE=0.05
STOP_LOSS=0.02
TAKE_PROFIT=0.04

# Risk Management
MAX_DRAWDOWN=0.3
EMERGENCY_STOP=0.5
COOLDOWN_PERIOD=300

# Data Sources
USE_LIVE_DATA=true
CACHE_DURATION=60
API_TIMEOUT=10

# Performance
BATCH_SIZE=50
UPDATE_INTERVAL=2
CLEANUP_INTERVAL=300
EOF

echo "✅ Created .env file with safe defaults"

# Set up shell environment for current session
echo "🔄 Setting up current shell environment..."

export TRADING_MODE=paper
export LOG_LEVEL=INFO
export OKX_API_KEY=paper_trading_key
export OKX_SECRET_KEY=paper_trading_secret
export OKX_PASSPHRASE=paper_trading_pass
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

echo "✅ Environment variables set for current session"

# Add to shell profile for persistence
SHELL_PROFILE=""
if [ -f ~/.zshrc ]; then
    SHELL_PROFILE=~/.zshrc
elif [ -f ~/.bashrc ]; then
    SHELL_PROFILE=~/.bashrc
elif [ -f ~/.bash_profile ]; then
    SHELL_PROFILE=~/.bash_profile
fi

if [ -n "$SHELL_PROFILE" ]; then
    echo "📋 Adding environment setup to $SHELL_PROFILE"
    
    # Remove any existing trading system env vars
    sed -i.bak '/# Trading System Environment/,/# End Trading System Environment/d' "$SHELL_PROFILE" 2>/dev/null
    
    # Add new environment setup
    cat >> "$SHELL_PROFILE" << 'EOF'

# Trading System Environment
export TRADING_MODE=paper
export LOG_LEVEL=INFO
export OKX_API_KEY=paper_trading_key
export OKX_SECRET_KEY=paper_trading_secret
export OKX_PASSPHRASE=paper_trading_pass
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
# End Trading System Environment
EOF
    
    echo "✅ Added to shell profile: $SHELL_PROFILE"
fi

echo ""
echo "🔍 CURRENT ENVIRONMENT STATUS:"
echo "=============================="
echo "Trading Mode: $TRADING_MODE"
echo "OKX API Key: ${OKX_API_KEY:0:10}..."
echo "PyTorch Config: $PYTORCH_CUDA_ALLOC_CONF"

echo ""
echo "✅ ENVIRONMENT SETUP COMPLETE!"
echo ""
echo "📋 What was configured:"
echo "  ✅ Paper trading mode (safe)"
echo "  ✅ Default API credentials for testing"
echo "  ✅ Performance optimizations"
echo "  ✅ Risk management defaults"
echo "  ✅ Shell profile integration"
echo ""
echo "🚀 READY TO TRADE!"
echo ""
echo "Next steps:"
echo "1. Test the system: python real_trading_system.py"
echo "2. For live trading: Update .env with real API keys"
echo "3. Change TRADING_MODE=live when ready"
echo ""
echo "⚠️  IMPORTANT NOTES:"
echo "• Currently in PAPER TRADING mode (safe)"
echo "• No real money at risk"
echo "• Update .env file for live trading"
echo "• Test thoroughly before going live"