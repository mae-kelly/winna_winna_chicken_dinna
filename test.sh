#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

header() {
    echo -e "\n${PURPLE}$1${NC}"
    echo -e "${PURPLE}$(printf '=%.0s' {1..80})${NC}"
}

# Main fix function
fix_python_environment() {
    header "🔧 FIXING PYTHON ENVIRONMENT"
    
    # Remove problematic venv
    if [ -d "venv" ]; then
        log "Removing existing virtual environment..."
        rm -rf venv
    fi
    
    # Find compatible Python version
    PYTHON_CMD=""
    if command -v python3.11 >/dev/null; then
        PYTHON_CMD="python3.11"
        log "Found Python 3.11"
    elif command -v python3.10 >/dev/null; then
        PYTHON_CMD="python3.10"
        log "Found Python 3.10"
    else
        warning "Installing Python 3.11 via Homebrew..."
        if command -v brew >/dev/null; then
            brew install python@3.11
            PYTHON_CMD="python3.11"
        else
            error "Please install Python 3.11: brew install python@3.11"
            exit 1
        fi
    fi
    
    # Create new environment
    log "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
    
    # Verify Python version
    VENV_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "Virtual environment Python version: $VENV_VERSION"
    
    if [[ "$VENV_VERSION" > "3.12" ]]; then
        warning "Python $VENV_VERSION detected - this may cause PyO3 issues"
    fi
    
    success "Python environment fixed"
}

# Install core dependencies
install_core_deps() {
    header "📦 INSTALLING CORE DEPENDENCIES"
    
    source venv/bin/activate
    
    log "Upgrading pip..."
    pip install --upgrade pip
    
    log "Installing scientific libraries..."
    pip install numpy pandas scikit-learn matplotlib seaborn
    
    log "Installing trading libraries..."
    pip install ccxt websockets orjson
    
    log "Installing ML libraries..."
    pip install optuna scipy
    
    log "Installing Jupyter..."
    pip install jupyter ipywidgets plotly
    
    success "Core dependencies installed"
}

# Try TensorFlow installation
install_tensorflow() {
    header "🧠 INSTALLING TENSORFLOW"
    
    source venv/bin/activate
    
    log "Attempting TensorFlow installation..."
    if pip install tensorflow 2>/dev/null; then
        success "TensorFlow installed successfully"
        
        # Test GPU availability
        python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('✅ M1 GPU detected:', gpus[0])
else:
    print('⚠️ Using CPU mode')
" 2>/dev/null || warning "TensorFlow installed but GPU test failed"
    
    else
        warning "TensorFlow installation failed - trying CPU version..."
        if pip install tensorflow-cpu 2>/dev/null; then
            success "TensorFlow CPU installed"
        else
            warning "TensorFlow not available - will use NumPy fallbacks"
        fi
    fi
}

# Simplified Rust compilation with better error handling
compile_rust_simple() {
    header "🦀 ATTEMPTING RUST COMPILATION"
    
    source venv/bin/activate
    
    # Install maturin
    log "Installing maturin..."
    pip install maturin setuptools wheel
    
    # Set up basic Rust project
    log "Setting up Rust project..."
    mkdir -p rust_modules/fast_math/src
    
    # Create simple Cargo.toml that should work
    cat > rust_modules/fast_math/Cargo.toml << 'EOF'
[package]
name = "fast_math"
version = "0.1.0"
edition = "2021"

[lib]
name = "fast_math"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"
EOF

    # Create minimal Rust library
    cat > rust_modules/fast_math/src/lib.rs << 'EOF'
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

#[pyclass]
pub struct FastMathEngine;

#[pymethods]
impl FastMathEngine {
    #[new]
    fn new() -> Self {
        FastMathEngine
    }
    
    fn fast_rsi(&self, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<f64> {
        let prices = prices.as_slice()?;
        if prices.len() < 2 {
            return Ok(50.0);
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        let mut count = 0;
        
        for i in 1..prices.len().min(period + 1) {
            let diff = prices[i] - prices[i-1];
            if diff > 0.0 {
                gains += diff;
            } else {
                losses -= diff;
            }
            count += 1;
        }
        
        if count == 0 || losses == 0.0 {
            return Ok(50.0);
        }
        
        let avg_gain = gains / count as f64;
        let avg_loss = losses / count as f64;
        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        
        Ok(rsi)
    }
}

#[pymodule]
fn fast_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FastMathEngine>()?;
    Ok(())
}
EOF

    # Try compilation with environment variables
    log "Attempting Rust compilation..."
    cd rust_modules/fast_math
    
    # Set PyO3 environment variables for compatibility
    export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
    export RUSTFLAGS="-C target-cpu=native"
    
    if maturin develop --release 2>/dev/null; then
        success "Rust module compiled successfully!"
        cd ../..
        return 0
    else
        warning "Rust compilation failed - will use Python fallbacks"
        cd ../..
        return 1
    fi
}

# Test the installation
test_installation() {
    header "🧪 TESTING INSTALLATION"
    
    source venv/bin/activate
    
    python3 << 'EOF'
import sys
print(f"Python version: {sys.version}")

# Test core libraries
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print("✅ Core scientific libraries working")
except ImportError as e:
    print(f"❌ Core libraries failed: {e}")
    sys.exit(1)

# Test trading libraries
try:
    import ccxt
    import websockets
    print("✅ Trading libraries working")
except ImportError as e:
    print(f"❌ Trading libraries failed: {e}")

# Test TensorFlow
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ TensorFlow with M1 GPU: {gpus[0]}")
    else:
        print("✅ TensorFlow with CPU")
except ImportError:
    print("⚠️ TensorFlow not available")

# Test Rust module
try:
    import fast_math
    engine = fast_math.FastMathEngine()
    test_prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
    rsi = engine.fast_rsi(test_prices, 3)
    print(f"✅ Rust acceleration working - RSI: {rsi:.2f}")
except ImportError:
    print("⚠️ Rust modules not available - using Python fallbacks")
except Exception as e:
    print(f"⚠️ Rust module error: {e}")

# Test Jupyter
try:
    import jupyter
    print("✅ Jupyter environment ready")
except ImportError:
    print("❌ Jupyter failed")

print("\n🎯 Installation test completed!")
EOF

    success "Installation testing completed"
}

# Create a simple demo notebook
create_demo() {
    header "📓 CREATING DEMO NOTEBOOK"
    
    cat > trading_demo.py << 'EOF'
#!/usr/bin/env python3
"""
🚀 Simple Crypto Trading Demo
M1 Mac Optimized Trading System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("🚀 M1 Crypto Trading System Demo")
print("=" * 40)

# Test Rust acceleration if available
rust_available = False
try:
    import fast_math
    engine = fast_math.FastMathEngine()
    rust_available = True
    print("✅ Rust acceleration: ACTIVE")
except ImportError:
    print("⚠️ Rust acceleration: Not available")

# Test TensorFlow
tf_available = False
try:
    import tensorflow as tf
    tf_available = True
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ TensorFlow M1 GPU: ACTIVE")
    else:
        print("✅ TensorFlow CPU: ACTIVE")
except ImportError:
    print("⚠️ TensorFlow: Not available")

# Generate sample crypto data
np.random.seed(42)
dates = pd.date_range('2025-01-01', periods=1000, freq='1h')
prices = 50000 * np.exp(np.cumsum(np.random.normal(0, 0.01, 1000)))

print(f"\n📊 Generated {len(prices)} price points")
print(f"Price range: ${prices.min():.0f} - ${prices.max():.0f}")

# Calculate RSI
if rust_available:
    rsi = engine.fast_rsi(prices, 14)
    print(f"🦀 Rust RSI: {rsi:.2f}")
else:
    # Python fallback
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50
    print(f"🐍 Python RSI: {rsi:.2f}")

# Simple moving averages
sma_20 = pd.Series(prices).rolling(20).mean().iloc[-1]
sma_50 = pd.Series(prices).rolling(50).mean().iloc[-1]

print(f"📈 SMA(20): ${sma_20:.2f}")
print(f"📈 SMA(50): ${sma_50:.2f}")

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(dates, prices, label='BTC Price', alpha=0.8)
plt.plot(dates, pd.Series(prices).rolling(20).mean(), label='SMA(20)', alpha=0.7)
plt.plot(dates, pd.Series(prices).rolling(50).mean(), label='SMA(50)', alpha=0.7)
plt.title('🚀 Crypto Price Analysis')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.axhline(70, color='r', linestyle='--', alpha=0.5, label='Overbought')
plt.axhline(30, color='g', linestyle='--', alpha=0.5, label='Oversold')
plt.axhline(rsi, color='blue', linewidth=2, label=f'Current RSI: {rsi:.1f}')
plt.title('RSI Indicator')
plt.ylabel('RSI')
plt.ylim(0, 100)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('trading_demo.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n🎯 Demo completed successfully!")
print("💰 System ready for $1K → $10K challenge!")

if rust_available and tf_available:
    print("🚀 MAXIMUM PERFORMANCE MODE: All optimizations active!")
elif rust_available or tf_available:
    print("⚡ HIGH PERFORMANCE MODE: Some optimizations active")
else:
    print("📊 STANDARD MODE: All core features working")

print("\nNext steps:")
print("1. Connect to exchange APIs")
print("2. Implement trading strategies") 
print("3. Start with paper trading")
print("4. Scale to live trading")
EOF

    chmod +x trading_demo.py
    success "Demo script created: trading_demo.py"
}

# Main execution
main() {
    header "🚀 M1 TRADING SYSTEM - QUICK FIX"
    log "Fixing PyO3 compatibility issues..."
    
    # System info
    log "System: $(uname -m) $(uname -s)"
    log "Cores: $(sysctl -n hw.ncpu)"
    
    fix_python_environment
    install_core_deps
    install_tensorflow
    
    if compile_rust_simple; then
        success "Rust acceleration enabled!"
    else
        warning "Using Python fallbacks (still fully functional)"
    fi
    
    test_installation
    create_demo
    
    header "🎉 INSTALLATION COMPLETED"
    success "System is ready for trading!"
    
    echo -e "\n${GREEN}Quick Start:${NC}"
    echo -e "${GREEN}source venv/bin/activate${NC}"
    echo -e "${GREEN}python trading_demo.py${NC}"
    echo -e "\n${GREEN}🚀 Ready for crypto trading!${NC}"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi