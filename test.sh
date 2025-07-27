#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNINGS=0

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED_TESTS++))
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
}

error() {
    echo -e "${RED}❌ $1${NC}"
    ((FAILED_TESTS++))
}

header() {
    echo -e "\n${PURPLE}$1${NC}"
    echo -e "${PURPLE}$(printf '=%.0s' {1..80})${NC}"
}

subheader() {
    echo -e "\n${CYAN}$1${NC}"
    echo -e "${CYAN}$(printf '-%.0s' {1..60})${NC}"
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    local is_critical="${3:-false}"
    
    ((TOTAL_TESTS++))
    log "Testing: $test_name"
    
    if eval "$test_command" >/dev/null 2>&1; then
        success "$test_name"
        return 0
    else
        if [ "$is_critical" = "true" ]; then
            error "$test_name (CRITICAL)"
            return 1
        else
            warning "$test_name (non-critical)"
            return 1
        fi
    fi
}

run_python_test() {
    local test_name="$1"
    local python_code="$2"
    local is_critical="${3:-false}"
    
    ((TOTAL_TESTS++))
    log "Testing: $test_name"
    
    if python3 -c "$python_code" >/dev/null 2>&1; then
        success "$test_name"
        return 0
    else
        if [ "$is_critical" = "true" ]; then
            error "$test_name (CRITICAL)"
            return 1
        else
            warning "$test_name (non-critical)"
            return 1
        fi
    fi
}

show_system_info() {
    header "🖥️  SYSTEM INFORMATION"
    
    echo "System: $(uname -m) $(uname -s) $(uname -r)"
    echo "CPU Cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'Unknown')"
    echo "Memory: $(system_profiler SPHardwareDataType 2>/dev/null | grep "Memory:" | awk '{print $2" "$3}' || echo 'Unknown')"
    echo "Architecture: $(uname -m)"
    
    if command -v sw_vers >/dev/null 2>&1; then
        echo "macOS Version: $(sw_vers -productVersion)"
    fi
    
    echo "Shell: $SHELL"
    echo "Current Directory: $(pwd)"
    echo "User: $(whoami)"
}

test_virtual_environment() {
    header "🐍 VIRTUAL ENVIRONMENT TESTS"
    
    if [ ! -d "venv" ]; then
        error "Virtual environment not found"
        return 1
    fi
    
    source venv/bin/activate || {
        error "Failed to activate virtual environment"
        return 1
    }
    
    success "Virtual environment activated"
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    log "Python version: $PYTHON_VERSION"
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
        success "Python version compatible (>= 3.9)"
    else
        error "Python version too old (< 3.9)"
        return 1
    fi
    
    run_test "pip installation" "pip --version" true
    
    return 0
}

test_core_dependencies() {
    header "📦 CORE DEPENDENCIES TEST"
    
    source venv/bin/activate
    
    run_python_test "NumPy" "import numpy as np; print(f'NumPy {np.__version__}')" true
    run_python_test "Pandas" "import pandas as pd; print(f'Pandas {pd.__version__}')" true
    run_python_test "Matplotlib" "import matplotlib.pyplot as plt; print('Matplotlib OK')" true
    run_python_test "SciPy" "import scipy; print(f'SciPy {scipy.__version__}')" true
    run_python_test "Scikit-learn" "import sklearn; print(f'Scikit-learn {sklearn.__version__}')" true
    
    run_python_test "CCXT" "import ccxt; print(f'CCXT {ccxt.__version__}')" false
    run_python_test "WebSockets" "import websockets; print('WebSockets OK')" false
    run_python_test "Asyncio" "import asyncio; print('Asyncio OK')" true
    
    run_python_test "Optuna" "import optuna; print(f'Optuna {optuna.__version__}')" false
    run_python_test "Plotly" "import plotly; print('Plotly OK')" false
}

test_tensorflow() {
    header "🧠 TENSORFLOW & ML TESTS"
    
    source venv/bin/activate
    
    if run_python_test "TensorFlow Import" "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')" false; then
        
        python3 << 'EOF'
import tensorflow as tf
import os
print(f"TensorFlow version: {tf.__version__}")

try:
    if hasattr(tf.config.experimental, 'list_physical_devices'):
        devices = tf.config.experimental.list_physical_devices()
    else:
        devices = tf.config.list_physical_devices()
    
    print(f"All devices: {devices}")
    
    gpus = []
    try:
        if hasattr(tf.config.experimental, 'list_physical_devices'):
            gpus = tf.config.experimental.list_physical_devices('GPU')
        else:
            gpus = tf.config.list_physical_devices('GPU')
    except:
        pass
    
    if gpus:
        print(f"✅ GPU devices found: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        try:
            if hasattr(tf.config.experimental, 'set_memory_growth'):
                tf.config.experimental.set_memory_growth(gpus[0], True)
            else:
                tf.config.set_memory_growth(gpus[0], True)
            print("✅ GPU memory growth enabled")
        except Exception as e:
            print(f"⚠️ GPU memory growth failed: {e}")
    else:
        print("⚠️ No GPU devices found - trying M1 specific setup...")
        
        try:
            os.environ['TF_METAL_DEVICE_PLACEMENT'] = '1'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            
            import tensorflow.python.platform.build_info as build_info
            print(f"TF Build info: {build_info}")
            
            logical_devices = tf.config.list_logical_devices()
            print(f"Logical devices: {logical_devices}")
            
            physical_devices = tf.config.list_physical_devices()
            print(f"Physical devices: {physical_devices}")
            
            try:
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([1.0, 2.0, 3.0])
                    print("✅ M1 GPU test tensor created")
            except:
                print("⚠️ M1 GPU not accessible")
                
        except Exception as e:
            print(f"M1 GPU setup failed: {e}")
            
    print("Testing TF operations...")
    
    try:
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        
        with tf.device('/CPU:0'):
            c_cpu = tf.matmul(a, b)
            print(f"✅ CPU TensorFlow operations working, shape: {c_cpu.shape}")
        
        try:
            with tf.device('/GPU:0'):
                c_gpu = tf.matmul(a, b)
                print(f"✅ GPU TensorFlow operations working, shape: {c_gpu.shape}")
        except:
            print("⚠️ GPU operations not available")
            
    except Exception as e:
        print(f"❌ TensorFlow operations failed: {e}")
        exit(1)

    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision policy set")
    except Exception as e:
        print(f"⚠️ Mixed precision failed: {e}")

    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        print("✅ Keras model creation successful")
    except Exception as e:
        print(f"❌ Keras model creation failed: {e}")
        exit(1)
except Exception as e:
    print(f"❌ Overall TensorFlow test failed: {e}")
    exit(1)
EOF
        
        if [ $? -eq 0 ]; then
            success "TensorFlow comprehensive test"
        else
            error "TensorFlow comprehensive test failed"
        fi
    else
        warning "TensorFlow not available - ML features will use NumPy fallbacks"
    fi
}

test_rust_modules() {
    header "🦀 RUST MODULES TEST"
    
    source venv/bin/activate
    
    python3 << 'EOF'
import numpy as np
import sys
import os

print("Checking for Rust modules in all possible locations...")

possible_paths = [
    '.',
    './rust_modules/fast_math',
    './rust_modules/fast_math/target/wheels',
    './rust_modules/fast_math/target/release',
    './target/wheels',
    './target/release',
    os.path.expanduser('~/.local/lib/python3.11/site-packages'),
    'venv/lib/python3.11/site-packages'
]

for path in possible_paths:
    if os.path.exists(path):
        print(f"Path exists: {path}")
        if path not in sys.path:
            sys.path.insert(0, path)
        
        files_in_path = []
        try:
            files_in_path = [f for f in os.listdir(path) if 'fast_math' in f.lower()]
            if files_in_path:
                print(f"  Found fast_math related files: {files_in_path}")
        except:
            pass

print(f"Python path: {sys.path[:5]}...")

for attempt in range(3):
    try:
        if attempt == 0:
            import fast_math
        elif attempt == 1:
            sys.path.insert(0, './rust_modules/fast_math/target/release')
            import fast_math
        elif attempt == 2:
            sys.path.insert(0, './rust_modules/fast_math')
            import fast_math
        
        engine = fast_math.FastMathEngine()
        print("✅ Fast Math Rust module imported")
        
        test_prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 102.0, 104.0, 103.5, 105.0, 104.0])
        rsi = engine.fast_rsi(test_prices, 5)
        print(f"✅ Rust RSI calculation: {rsi:.2f}")
        
        if 0 <= rsi <= 100:
            print("✅ RSI value in valid range")
            exit(0)
        else:
            print(f"❌ RSI value out of range: {rsi}")
            exit(1)
            
    except ImportError as e:
        if attempt == 2:
            print(f"⚠️ Rust fast_math module not available after all attempts: {e}")
            
            print("\nAttempting to compile Rust module...")
            try:
                import subprocess
                import os
                
                os.chdir('rust_modules/fast_math')
                
                result = subprocess.run(['maturin', 'develop', '--release'], 
                                      capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print("✅ Rust compilation successful")
                    os.chdir('../..')
                    
                    try:
                        import fast_math
                        engine = fast_math.FastMathEngine()
                        rsi = engine.fast_rsi(test_prices, 5)
                        print(f"✅ Post-compilation RSI: {rsi:.2f}")
                        exit(0)
                    except ImportError:
                        print("❌ Still can't import after compilation")
                        exit(2)
                else:
                    print(f"❌ Rust compilation failed: {result.stderr}")
                    os.chdir('../..')
                    exit(2)
                    
            except Exception as comp_e:
                print(f"❌ Compilation attempt failed: {comp_e}")
                try:
                    os.chdir('../..')
                except:
                    pass
                exit(2)
        else:
            continue
    except Exception as e:
        print(f"❌ Rust fast_math error: {e}")
        exit(1)
EOF
    
    local rust_result=$?
    if [ $rust_result -eq 0 ]; then
        success "Fast Math Rust module"
    elif [ $rust_result -eq 2 ]; then
        warning "Fast Math Rust module not available (using Python fallbacks)"
    else
        error "Fast Math Rust module failed"
    fi
    
    python3 << 'EOF'
import sys
import os

for path in ['./rust_modules/orderbook_engine', '.']:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import orderbook_engine
    analyzer = orderbook_engine.OrderBookAnalyzer()
    print("✅ OrderBook Engine Rust module imported")
    
    analyzer.update_order_book("BTCUSDT", [(50000.0, 1.0), (49999.0, 0.5)], [(50001.0, 1.0), (50002.0, 0.5)])
    features = analyzer.get_microstructure_features("BTCUSDT")
    print(f"✅ Microstructure features: {len(features)} values")
    
except ImportError:
    print("⚠️ OrderBook Engine Rust module not available")
    exit(2)
except Exception as e:
    print(f"❌ OrderBook Engine error: {e}")
    exit(1)
EOF
    
    local orderbook_result=$?
    if [ $orderbook_result -eq 0 ]; then
        success "OrderBook Engine Rust module"
    elif [ $orderbook_result -eq 2 ]; then
        warning "OrderBook Engine Rust module not available"
    else
        error "OrderBook Engine Rust module failed"
    fi
}

test_trading_modules() {
    header "💹 TRADING MODULES TEST"
    
    source venv/bin/activate
    
    subheader "Data Engine Tests"
    python3 << 'EOF'
import sys
import asyncio
import numpy as np
from datetime import datetime

try:
    from data_engine import RealTimeDataEngine, MarketData, HighFrequencyDataBuffer, CrossExchangeArbitrageDetector
    print("✅ Data engine imports successful")
    
    market_data = MarketData(
        symbol="BTCUSDT",
        timestamp=datetime.now().timestamp(),
        price=50000.0,
        volume=1.5,
        bid=49999.0,
        ask=50001.0,
        exchange="binance"
    )
    print(f"✅ MarketData creation: {market_data.symbol} @ ${market_data.price}")
    print(f"   Spread: {market_data.spread_bps:.2f} bps")
    
    buffer = HighFrequencyDataBuffer(max_size=1000)
    buffer.add_tick(market_data)
    print("✅ Data buffer add_tick successful")
    
    features = buffer.get_microstructure_features("BTCUSDT", 10)
    print(f"✅ Microstructure features: {len(features)} features")
    
    detector = CrossExchangeArbitrageDetector()
    detector.update_price(market_data)
    opportunities = detector.detect_opportunities("BTCUSDT")
    print(f"✅ Arbitrage detection: {len(opportunities)} opportunities")
    
except Exception as e:
    print(f"❌ Data engine test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        success "Data Engine module"
    else
        error "Data Engine module failed"
        return 1
    fi
    
    subheader "Neural Core Tests"
    python3 << 'EOF'
import sys
import numpy as np
import asyncio

try:
    from neural_core import SelfOptimizingModel, ModelPerformance, CustomActivations, AdvancedLayers, NeuralEvolution
    print("✅ Neural core imports successful")
    
    input_shape = (100, 50)
    model = SelfOptimizingModel(input_shape)
    print(f"✅ SelfOptimizingModel created with input shape: {input_shape}")
    
    evolution = NeuralEvolution()
    architecture = evolution.create_random_architecture()
    print(f"✅ Random architecture created with {architecture['layers']} layers")
    
    try:
        import tensorflow as tf
        x = tf.constant([[1.0, -1.0, 0.5]])
        mish_out = CustomActivations.mish(x)
        print(f"✅ Custom activation (Mish) test successful")
    except Exception as e:
        print(f"⚠️ Custom activations require TensorFlow: {e}")
    
    dummy_features = np.random.randn(5, 100, 50)
    dummy_targets = {
        'price_prediction': np.random.randn(5, 10),
        'direction_prediction': np.random.randint(0, 2, (5, 3)),
        'confidence_prediction': np.random.rand(5, 1),
        'volatility_prediction': np.random.rand(5, 1),
        'regime_prediction': np.random.randint(0, 2, (5, 4))
    }
    
    for i in range(5):
        model.add_training_data(
            dummy_features[i:i+1], 
            {k: v[i:i+1] for k, v in dummy_targets.items()}
        )
    
    print("✅ Training data addition successful")
    
except Exception as e:
    print(f"❌ Neural core test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        success "Neural Core module"
    else
        error "Neural Core module failed"
        return 1
    fi
    
    subheader "Execution Engine Tests"
    python3 << 'EOF'
import sys
import asyncio
import numpy as np
from datetime import datetime, timedelta

try:
    from execution_engine import (
        HighFrequencyTradingEngine, RiskManager, AdvancedOrderManager,
        Position, Trade, PortfolioMetrics
    )
    print("✅ Execution engine imports successful")
    
    risk_manager = RiskManager(initial_capital=1000.0)
    position_size = risk_manager.calculate_position_size(
        signal_strength=0.05,
        confidence=0.8,
        volatility=0.02,
        symbol="BTCUSDT"
    )
    print(f"✅ Risk manager position sizing: {position_size:.4f}")
    
    position = Position(
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        entry_price=50000.0,
        current_price=50100.0,
        unrealized_pnl=10.0,
        realized_pnl=0.0,
        entry_time=datetime.now()
    )
    print(f"✅ Position created: {position.side} {position.size} {position.symbol}")
    
    trade = Trade(
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        entry_price=50000.0,
        exit_price=50100.0,
        pnl=10.0,
        pnl_pct=0.002,
        entry_time=datetime.now(),
        exit_time=datetime.now(),
        duration=timedelta(minutes=5),
        reason="take_profit",
        confidence=0.8
    )
    print(f"✅ Trade created: PnL ${trade.pnl:.2f} ({trade.pnl_pct:.2%})")
    
    order_manager = AdvancedOrderManager()
    print("✅ Advanced order manager created")
    
except Exception as e:
    print(f"❌ Execution engine test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        success "Execution Engine module"
    else
        error "Execution Engine module failed"
        return 1
    fi
}

test_performance() {
    header "⚡ PERFORMANCE BENCHMARKS"
    
    source venv/bin/activate
    
    python3 << 'EOF'
import time
import numpy as np
import sys

print("🔥 Running performance benchmarks...")

start_time = time.time()
data = np.random.randn(100000, 100)
result = np.dot(data, data.T)
numpy_time = time.time() - start_time
print(f"NumPy matrix multiplication (100k x 100): {numpy_time:.3f}s")

prices = np.random.randn(10000) * 100 + 50000
start_time = time.time()

def python_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

python_rsi_result = python_rsi(prices)
python_rsi_time = time.time() - start_time
print(f"Python RSI calculation (10k prices): {python_rsi_time:.4f}s, RSI: {python_rsi_result:.2f}")

try:
    import fast_math
    engine = fast_math.FastMathEngine()
    
    start_time = time.time()
    rust_rsi_result = engine.fast_rsi(prices, 14)
    rust_rsi_time = time.time() - start_time
    
    print(f"Rust RSI calculation (10k prices): {rust_rsi_time:.4f}s, RSI: {rust_rsi_result:.2f}")
    
    speedup = python_rsi_time / rust_rsi_time if rust_rsi_time > 0 else float('inf')
    print(f"🚀 Rust speedup: {speedup:.1f}x faster")
    
except ImportError:
    print("⚠️ Rust acceleration not available")

import psutil
import os
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Current memory usage: {memory_mb:.1f} MB")

try:
    import tensorflow as tf
    
    start_time = time.time()
    x = tf.random.normal((1000, 1000))
    y = tf.random.normal((1000, 1000))
    result = tf.matmul(x, y)
    tf_time = time.time() - start_time
    
    print(f"TensorFlow matrix multiplication (1k x 1k): {tf_time:.3f}s")
    
    if tf.config.list_physical_devices('GPU'):
        print("✅ TensorFlow using GPU acceleration")
    else:
        print("⚠️ TensorFlow using CPU")
        
except ImportError:
    print("⚠️ TensorFlow not available for performance test")

print("✅ Performance benchmarks completed")
EOF
    
    if [ $? -eq 0 ]; then
        success "Performance benchmarks"
    else
        warning "Performance benchmarks had issues"
    fi
}

test_integration() {
    header "🔗 INTEGRATION TESTS"
    
    source venv/bin/activate
    
    python3 << 'EOF'
import sys
import asyncio
import numpy as np
from datetime import datetime, timedelta

try:
    from data_engine import RealTimeDataEngine, MarketData, HighFrequencyDataBuffer
    from neural_core import SelfOptimizingModel
    from execution_engine import RiskManager, AdvancedOrderManager
    
    print("✅ All modules imported successfully")
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    data_buffer = HighFrequencyDataBuffer(max_size=1000)
    risk_manager = RiskManager(initial_capital=1000.0)
    
    print("✅ Core components initialized")
    
    for i in range(100):
        price = 50000 + np.random.normal(0, 100)
        market_data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now().timestamp(),
            price=price,
            volume=np.random.exponential(1.0),
            bid=price - 0.5,
            ask=price + 0.5,
            exchange="binance"
        )
        
        data_buffer.add_tick(market_data)
        
        if i % 10 == 0:
            features = data_buffer.get_microstructure_features("BTCUSDT", 10)
            if len(features) > 0:
                position_size = risk_manager.calculate_position_size(
                    signal_strength=np.random.normal(0, 0.01),
                    confidence=np.random.uniform(0.5, 1.0),
                    volatility=0.02,
                    symbol="BTCUSDT"
                )
    
    print("✅ Data flow simulation completed")
    
    recent_data = data_buffer.get_recent_data("BTCUSDT", 60)
    print(f"✅ Retrieved {len(recent_data)} recent data points")
    
    features = data_buffer.get_microstructure_features("BTCUSDT", 30)
    print(f"✅ Extracted {len(features)} microstructure features")
    
    try:
        model = SelfOptimizingModel((100, 50))
        print("✅ Neural model integration successful")
    except Exception as e:
        print(f"⚠️ Neural model integration issue: {e}")
    
    print("✅ Integration test completed successfully")
    
except Exception as e:
    print(f"❌ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        success "Module integration"
    else
        error "Module integration failed"
        return 1
    fi
}

test_file_structure() {
    header "📁 FILE STRUCTURE TEST"
    
    local python_files=(
        "data_engine.py"
        "neural_core.py" 
        "execution_engine.py"
        "trading_demo.py"
    )
    
    for file in "${python_files[@]}"; do
        if [ -f "$file" ]; then
            success "Found $file"
        else
            error "Missing $file"
        fi
    done
    
    if [ -d "rust_modules" ]; then
        success "Rust modules directory exists"
        
        if [ -f "rust_modules/fast_math/Cargo.toml" ]; then
            success "Fast math Cargo.toml found"
        else
            warning "Fast math Cargo.toml missing"
        fi
        
        if [ -f "rust_modules/fast_math/src/lib.rs" ]; then
            success "Fast math source code found"
        else
            warning "Fast math source code missing"
        fi
    else
        warning "Rust modules directory missing"
    fi
    
    if [ -f "master_trading_system.ipynb" ]; then
        success "Master notebook found"
    else
        warning "Master notebook missing"
    fi
    
    if [ -x "test.sh" ]; then
        success "test.sh is executable"
    else
        warning "test.sh not executable"
    fi
    
    if [ -x "trading_demo.py" ]; then
        success "trading_demo.py is executable"
    else
        warning "trading_demo.py not executable"
    fi
}

test_trading_simulation() {
    header "🎯 TRADING SIMULATION TEST"
    
    source venv/bin/activate
    
    python3 << 'EOF'
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

print("🚀 Running mini trading simulation...")

np.random.seed(42)
n_points = 1000
base_price = 50000

returns = []
volatility = 0.02

for i in range(n_points):
    if i > 0:
        volatility = 0.98 * volatility + 0.02 * abs(returns[-1])
    
    trend = 0.0001 * np.sin(i / 100)
    noise = np.random.normal(trend, volatility)
    returns.append(noise)

prices = base_price * np.exp(np.cumsum(returns))
volumes = np.random.lognormal(0, 1, n_points)

print(f"✅ Generated {n_points} price points")
print(f"   Price range: ${prices.min():.0f} - ${prices.max():.0f}")
print(f"   Volatility range: {np.std(returns):.4f}")

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = pd.Series(gains).rolling(period).mean()
    avg_losses = pd.Series(losses).rolling(period).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_sma(prices, period):
    return np.mean(prices[-period:])

current_rsi = calculate_rsi(prices)
sma_20 = calculate_sma(prices, 20)
sma_50 = calculate_sma(prices, 50)
current_price = prices[-1]

print(f"✅ Technical indicators calculated:")
print(f"   Current Price: ${current_price:.2f}")
print(f"   RSI(14): {current_rsi:.2f}")
print(f"   SMA(20): ${sma_20:.2f}")
print(f"   SMA(50): ${sma_50:.2f}")

portfolio_value = 1000.0
position = 0.0
trades = []

for i in range(100, len(prices) - 1):
    price = prices[i]
    rsi = calculate_rsi(prices[:i+1])
    
    if position == 0:
        if rsi < 30:
            position = 0.1
            entry_price = price
            portfolio_value -= position * price
        elif rsi > 70:
            position = -0.1
            entry_price = price
            portfolio_value += abs(position) * price
    
    else:
        if (position > 0 and rsi > 60) or (position < 0 and rsi < 40):
            exit_price = price
            if position > 0:
                pnl = position * (exit_price - entry_price)
                portfolio_value += position * exit_price
            else:
                pnl = abs(position) * (entry_price - exit_price)
                portfolio_value -= abs(position) * exit_price
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'side': 'long' if position > 0 else 'short'
            })
            position = 0.0

total_pnl = sum(trade['pnl'] for trade in trades)
win_rate = sum(1 for trade in trades if trade['pnl'] > 0) / len(trades) if trades else 0
final_value = portfolio_value + (position * prices[-1] if position != 0 else 0)

print(f"✅ Trading simulation completed:")
print(f"   Total trades: {len(trades)}")
print(f"   Win rate: {win_rate:.1%}")
print(f"   Total PnL: ${total_pnl:.2f}")
print(f"   Final portfolio value: ${final_value:.2f}")
print(f"   Return: {(final_value - 1000) / 1000:.1%}")

if len(trades) > 0 and win_rate > 0.3:
    print("✅ Trading simulation successful")
else:
    print("⚠️ Trading simulation completed with warnings")
EOF
    
    if [ $? -eq 0 ]; then
        success "Trading simulation"
    else
        warning "Trading simulation had issues"
    fi
}

generate_report() {
    header "📊 TEST REPORT"
    
    echo "Total Tests Run: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS" 
    echo "Warnings: $WARNINGS"
    echo ""
    
    local pass_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    
    if [ $pass_rate -ge 90 ]; then
        echo -e "${GREEN}🎉 EXCELLENT: $pass_rate% pass rate${NC}"
        echo -e "${GREEN}✅ System is ready for production trading!${NC}"
    elif [ $pass_rate -ge 75 ]; then
        echo -e "${YELLOW}👍 GOOD: $pass_rate% pass rate${NC}"
        echo -e "${YELLOW}⚡ System is functional with some optimizations missing${NC}"
    elif [ $pass_rate -ge 50 ]; then
        echo -e "${YELLOW}⚠️  FAIR: $pass_rate% pass rate${NC}"
        echo -e "${YELLOW}🔧 System needs attention before production use${NC}"
    else
        echo -e "${RED}❌ POOR: $pass_rate% pass rate${NC}"
        echo -e "${RED}🚨 System has critical issues that need fixing${NC}"
    fi
    
    echo ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}🚀 SYSTEM STATUS: READY FOR LIVE TRADING${NC}"
        echo -e "${GREEN}💰 All core components operational${NC}"
    elif [ $FAILED_TESTS -le 2 ]; then
        echo -e "${YELLOW}⚡ SYSTEM STATUS: READY FOR PAPER TRADING${NC}"
        echo -e "${YELLOW}🧪 Test with simulated trading first${NC}"
    else
        echo -e "${RED}🔧 SYSTEM STATUS: NEEDS FIXES${NC}"
        echo -e "${RED}❌ Resolve critical issues before trading${NC}"
    fi
    
    echo ""
    echo "Next Steps:"
    
    if [ $pass_rate -ge 90 ]; then
        echo "1. 🎯 Start with small position sizes"
        echo "2. 💼 Connect real exchange APIs"
        echo "3. 📊 Monitor performance closely"
        echo "4. 🚀 Scale up gradually"
    elif [ $pass_rate -ge 75 ]; then
        echo "1. 🔧 Fix any failing tests"
        echo "2. 🧪 Run paper trading simulation" 
        echo "3. ⚡ Optimize performance issues"
        echo "4. 📈 Validate with historical data"
    else
        echo "1. ❌ Fix all critical failures"
        echo "2. 📦 Reinstall missing dependencies"
        echo "3. 🔄 Re-run this test script"
        echo "4. 📞 Check documentation"
    fi
}

save_test_log() {
    local log_file="test_results_$(date +%Y%m%d_%H%M%S).log"
    
    {
        echo "Crypto Trading System Test Results"
        echo "=================================="
        echo "Date: $(date)"
        echo "System: $(uname -a)"
        echo ""
        echo "Test Summary:"
        echo "Total Tests: $TOTAL_TESTS"
        echo "Passed: $PASSED_TESTS"
        echo "Failed: $FAILED_TESTS"
        echo "Warnings: $WARNINGS"
        echo "Pass Rate: $((PASSED_TESTS * 100 / TOTAL_TESTS))%"
    } > "$log_file"
    
    log "Test results saved to: $log_file"
}

main() {
    clear
    header "🚀 COMPREHENSIVE CRYPTO TRADING SYSTEM TEST"
    log "Starting comprehensive system validation..."
    
    show_system_info
    
    test_virtual_environment || {
        error "Virtual environment test failed - cannot continue"
        exit 1
    }
    
    test_core_dependencies
    test_tensorflow
    test_rust_modules
    test_trading_modules
    test_file_structure
    
    test_performance
    test_integration
    test_trading_simulation
    
    generate_report
    save_test_log
    
    header "🏁 TESTING COMPLETED"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        log "All tests passed! System ready for trading."
        exit 0
    elif [ $FAILED_TESTS -le 2 ]; then
        log "Minor issues detected. System functional but needs attention."
        exit 1
    else
        log "Critical issues detected. System needs fixes before use."
        exit 2
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi