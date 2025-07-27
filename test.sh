#!/bin/bash

echo "🚀 FINAL SETUP FIX FOR M1 TRADING SYSTEM"
echo "========================================"

# Deactivate any existing virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "🔄 Deactivating current environment..."
    deactivate 2>/dev/null || true
fi

# Remove any broken virtual environment
echo "🗑️  Removing broken virtual environment..."
rm -rf venv/ 2>/dev/null || true

# Create fresh virtual environment
echo "📦 Creating fresh virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Verify we're in the virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Successfully activated virtual environment: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install all required packages
echo "📥 Installing PyTorch with M1 GPU support..."
pip install torch torchvision torchaudio

echo "📦 Installing core dependencies..."
pip install numpy pandas matplotlib psutil orjson websockets aiohttp requests

echo "💹 Installing trading libraries..."
pip install python-binance ccxt scikit-learn python-dotenv

echo "🔴 Installing Redis..."
pip install redis aioredis==2.0.0

# Test M1 GPU availability
echo ""
echo "🧪 Testing M1 GPU..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'M1 GPU (MPS) available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print('✅ M1 GPU is ready!')
    x = torch.randn(10, device='mps')
    print(f'✅ M1 GPU test successful: {x.shape}')
else:
    print('❌ M1 GPU not available')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🧪 Now test your M1 system:"
echo "  python test_system_m1.py test"
echo ""
echo "💡 Make sure you're in the virtual environment with:"
echo "  source venv/bin/activate"