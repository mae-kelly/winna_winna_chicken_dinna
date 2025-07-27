#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
COMPRESSED_ARCHIVE="trading_system_optimized_$(date +%Y%m%d_%H%M%S).tar.gz"
TEMP_DIR="temp_optimize"

# Counters
DELETED_FILES=0
DELETED_DIRS=0
COMPRESSED_FILES=0
BYTES_SAVED=0

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
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

subheader() {
    echo -e "\n${CYAN}$1${NC}"
    echo -e "${CYAN}$(printf '-%.0s' {1..60})${NC}"
}

get_file_size() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        stat -f%z "$1" 2>/dev/null || echo 0
    else
        stat -c%s "$1" 2>/dev/null || echo 0
    fi
}

get_dir_size() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        du -sk "$1" 2>/dev/null | cut -f1 || echo 0
    else
        du -sk "$1" 2>/dev/null | cut -f1 || echo 0
    fi
}

format_bytes() {
    local bytes=$1
    if [ $bytes -ge 1073741824 ]; then
        echo "$(echo "scale=2; $bytes/1073741824" | bc)GB"
    elif [ $bytes -ge 1048576 ]; then
        echo "$(echo "scale=2; $bytes/1048576" | bc)MB"
    elif [ $bytes -ge 1024 ]; then
        echo "$(echo "scale=2; $bytes/1024" | bc)KB"
    else
        echo "${bytes}B"
    fi
}

safe_remove() {
    local target="$1"
    local description="$2"
    
    if [ -e "$target" ]; then
        local size_before=$(get_file_size "$target")
        if [ -d "$target" ]; then
            size_before=$(get_dir_size "$target")
            size_before=$((size_before * 1024))
        fi
        
        rm -rf "$target"
        BYTES_SAVED=$((BYTES_SAVED + size_before))
        
        if [ -d "$target" ] || [ -f "$target" ]; then
            ((DELETED_DIRS++))
        else
            ((DELETED_FILES++))
        fi
        
        log "Removed $description ($(format_bytes $size_before))"
    fi
}

create_backup() {
    header "📦 CREATING BACKUP"
    
    if [ ! -d "$BACKUP_DIR" ]; then
        mkdir -p "$BACKUP_DIR"
        success "Created backup directory: $BACKUP_DIR"
    fi
    
    # Backup critical files
    local critical_files=(
        "data_engine.py"
        "neural_core.py" 
        "execution_engine.py"
        "fast_math.rs"
        "orderbook_engine.rs"
        "master_trading_system.ipynb"
        "live_paper_trading.py"
        "trading_demo.py"
        "test.sh"
        ".gitignore"
    )
    
    for file in "${critical_files[@]}"; do
        if [ -f "$file" ]; then
            cp "$file" "$BACKUP_DIR/"
            log "Backed up: $file"
        fi
    done
    
    # Backup Rust modules structure
    if [ -d "rust_modules" ]; then
        cp -r "rust_modules" "$BACKUP_DIR/"
        log "Backed up: rust_modules/"
    fi
    
    success "Backup completed in $BACKUP_DIR"
}

clean_python_cache() {
    header "🐍 CLEANING PYTHON CACHE"
    
    # Remove __pycache__ directories
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Remove .pyc files
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove .pyo files
    find . -name "*.pyo" -delete 2>/dev/null || true
    
    # Remove .pytest_cache
    safe_remove ".pytest_cache" "pytest cache"
    
    # Remove .coverage files
    find . -name ".coverage*" -delete 2>/dev/null || true
    
    success "Python cache cleaned"
}

clean_rust_artifacts() {
    header "🦀 CLEANING RUST ARTIFACTS"
    
    # Clean Rust target directories
    find . -name "target" -type d | while read target_dir; do
        if [[ "$target_dir" == *"rust_modules"* ]]; then
            safe_remove "$target_dir" "Rust target directory"
        fi
    done
    
    # Remove Cargo.lock files (keep Cargo.toml)
    find . -name "Cargo.lock" -delete 2>/dev/null || true
    
    # Remove .rustc_info.json
    find . -name ".rustc_info.json" -delete 2>/dev/null || true
    
    # Remove CACHEDIR.TAG
    find . -name "CACHEDIR.TAG" -delete 2>/dev/null || true
    
    success "Rust artifacts cleaned"
}

clean_system_files() {
    header "🗂️  CLEANING SYSTEM FILES"
    
    # macOS files
    find . -name ".DS_Store" -delete 2>/dev/null || true
    find . -name "._*" -delete 2>/dev/null || true
    find . -name ".AppleDouble" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".LSOverride" -delete 2>/dev/null || true
    
    # Windows files
    find . -name "Thumbs.db" -delete 2>/dev/null || true
    find . -name "Desktop.ini" -delete 2>/dev/null || true
    find . -name "*.tmp" -delete 2>/dev/null || true
    
    # Linux files
    find . -name "*~" -delete 2>/dev/null || true
    find . -name ".nfs*" -delete 2>/dev/null || true
    
    # Editor files
    find . -name ".vscode" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".idea" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.swp" -delete 2>/dev/null || true
    find . -name "*.swo" -delete 2>/dev/null || true
    find . -name "*~" -delete 2>/dev/null || true
    
    success "System files cleaned"
}

clean_logs_temp() {
    header "📋 CLEANING LOGS & TEMP FILES"
    
    # Remove log files
    find . -name "*.log" -delete 2>/dev/null || true
    find . -name "*.out" -delete 2>/dev/null || true
    find . -name "*.err" -delete 2>/dev/null || true
    
    # Remove temporary files
    safe_remove "temp" "temp directory"
    safe_remove "tmp" "tmp directory"
    safe_remove ".tmp" "hidden tmp directory"
    
    # Remove test artifacts
    find . -name "test_results_*.log" -delete 2>/dev/null || true
    find . -name "trading_demo.png" -delete 2>/dev/null || true
    
    # Remove model files (these can be regenerated)
    find . -name "*.h5" -delete 2>/dev/null || true
    find . -name "*.pkl" -delete 2>/dev/null || true
    find . -name "*.joblib" -delete 2>/dev/null || true
    
    success "Logs and temp files cleaned"
}

optimize_python_files() {
    header "🔧 OPTIMIZING PYTHON FILES"
    
    # Remove docstrings and comments from non-critical Python files
    local files_to_optimize=(
        "data_engine.py"
        "neural_core.py"
        "execution_engine.py"
    )
    
    for file in "${files_to_optimize[@]}"; do
        if [ -f "$file" ]; then
            local size_before=$(get_file_size "$file")
            
            # Create optimized version (remove excessive whitespace)
            python3 << EOF
import re
import sys

def optimize_python_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Remove excessive blank lines (keep max 2 consecutive)
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # Remove trailing whitespace
    lines = []
    for line in content.split('\n'):
        lines.append(line.rstrip())
    
    content = '\n'.join(lines)
    
    # Remove empty lines at end of file
    content = content.rstrip() + '\n'
    
    with open(filename, 'w') as f:
        f.write(content)

optimize_python_file('$file')
EOF
            
            local size_after=$(get_file_size "$file")
            local saved=$((size_before - size_after))
            BYTES_SAVED=$((BYTES_SAVED + saved))
            
            log "Optimized $file (saved $(format_bytes $saved))"
        fi
    done
    
    success "Python files optimized"
}

compress_rust_source() {
    header "📦 COMPRESSING RUST SOURCE"
    
    if [ -d "rust_modules" ]; then
        # Create compressed archive of Rust source
        tar -czf "rust_modules_source.tar.gz" \
            --exclude="target" \
            --exclude="Cargo.lock" \
            --exclude=".rustc_info.json" \
            rust_modules/ 2>/dev/null || true
        
        if [ -f "rust_modules_source.tar.gz" ]; then
            local compressed_size=$(get_file_size "rust_modules_source.tar.gz")
            log "Created compressed Rust source: $(format_bytes $compressed_size)"
            ((COMPRESSED_FILES++))
        fi
    fi
    
    success "Rust source compressed"
}

optimize_notebook() {
    header "📓 OPTIMIZING NOTEBOOK"
    
    if [ -f "master_trading_system.ipynb" ]; then
        local size_before=$(get_file_size "master_trading_system.ipynb")
        
        # Remove output cells and metadata from notebook
        python3 << 'EOF'
import json
import sys

try:
    with open('master_trading_system.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Clear outputs and execution counts
    for cell in notebook.get('cells', []):
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
    
    # Remove metadata
    if 'metadata' in notebook:
        notebook['metadata'] = {
            'kernelspec': notebook['metadata'].get('kernelspec', {}),
            'language_info': notebook['metadata'].get('language_info', {})
        }
    
    with open('master_trading_system.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1, separators=(',', ':'))
    
    print("Notebook optimized")
except Exception as e:
    print(f"Error optimizing notebook: {e}")
EOF
        
        local size_after=$(get_file_size "master_trading_system.ipynb")
        local saved=$((size_before - size_after))
        BYTES_SAVED=$((BYTES_SAVED + saved))
        
        log "Optimized notebook (saved $(format_bytes $saved))"
    fi
    
    success "Notebook optimized"
}

create_minimal_requirements() {
    header "📋 CREATING MINIMAL REQUIREMENTS"
    
    cat > requirements_minimal.txt << 'EOF'
# Core Trading System Dependencies - Minimal Version
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.6.0
scikit-learn>=1.2.0

# Optional ML (install if needed)
# tensorflow>=2.13.0
# optuna>=3.2.0

# Optional Trading (install if needed)
# ccxt>=4.0.0
# websockets>=11.0.0

# Development (install if needed)
# jupyter>=1.0.0
# plotly>=5.0.0
EOF
    
    success "Created minimal requirements file"
}

create_install_script() {
    header "🚀 CREATING OPTIMIZED INSTALL SCRIPT"
    
    cat > install_optimized.sh << 'EOF'
#!/bin/bash

echo "🚀 Installing Optimized Crypto Trading System"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install minimal dependencies
pip install --upgrade pip
pip install -r requirements_minimal.txt

# Extract and compile Rust modules (if available)
if [ -f "rust_modules_source.tar.gz" ]; then
    echo "📦 Extracting Rust modules..."
    tar -xzf rust_modules_source.tar.gz
    
    # Install Rust if not available
    if ! command -v rustc &> /dev/null; then
        echo "🦀 Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
    fi
    
    # Install maturin for Python-Rust bindings
    pip install maturin
    
    # Compile Rust modules
    cd rust_modules/fast_math && maturin develop --release && cd ../..
    cd rust_modules/orderbook_engine && maturin develop --release && cd ../..
    
    echo "✅ Rust modules compiled and installed"
else
    echo "⚠️ Rust modules not available - using Python fallbacks"
fi

echo "✅ Installation completed!"
echo "Run: source venv/bin/activate && python trading_demo.py"
EOF
    
    chmod +x install_optimized.sh
    success "Created optimized install script"
}

create_optimized_archive() {
    header "📦 CREATING OPTIMIZED ARCHIVE"
    
    # Create temporary directory for final archive
    mkdir -p "$TEMP_DIR"
    
    # Copy essential files
    local essential_files=(
        "data_engine.py"
        "neural_core.py"
        "execution_engine.py" 
        "live_paper_trading.py"
        "trading_demo.py"
        "test.sh"
        "master_trading_system.ipynb"
        "requirements_minimal.txt"
        "install_optimized.sh"
        ".gitignore"
    )
    
    for file in "${essential_files[@]}"; do
        if [ -f "$file" ]; then
            cp "$file" "$TEMP_DIR/"
        fi
    done
    
    # Copy compressed Rust source if available
    if [ -f "rust_modules_source.tar.gz" ]; then
        cp "rust_modules_source.tar.gz" "$TEMP_DIR/"
    fi
    
    # Create README for optimized version
    cat > "$TEMP_DIR/README_OPTIMIZED.md" << 'EOF'
# Optimized Crypto Trading System

This is an optimized version of the advanced crypto trading system.

## Quick Start

1. Run the installation script:
   ```bash
   chmod +x install_optimized.sh
   ./install_optimized.sh
   ```

2. Activate environment and test:
   ```bash
   source venv/bin/activate
   python trading_demo.py
   ```

3. Run comprehensive tests:
   ```bash
   ./test.sh
   ```

## Features

- ⚡ Rust-accelerated mathematical operations
- 🧠 Self-optimizing neural networks
- 📊 Real-time data processing
- 💹 Advanced execution engine
- 🛡️ Comprehensive risk management

## File Structure

- `data_engine.py` - Real-time market data processing
- `neural_core.py` - AI/ML trading models
- `execution_engine.py` - Trade execution and portfolio management
- `trading_demo.py` - Simple demo and system test
- `test.sh` - Comprehensive system validation
- `rust_modules_source.tar.gz` - Compressed Rust acceleration modules

## System Requirements

- Python 3.9+
- 4GB+ RAM
- Optional: Rust compiler for acceleration
- Optional: CUDA-compatible GPU for ML acceleration

## Target Performance

- 🎯 Goal: $1,000 → $10,000 (10x return)
- ⚡ Ultra-low latency execution
- 🧠 Adaptive AI-driven strategies
- 📊 Multi-exchange arbitrage detection

Built for maximum performance and minimal resource usage.
EOF
    
    # Create the final compressed archive
    tar -czf "$COMPRESSED_ARCHIVE" -C "$TEMP_DIR" .
    
    local archive_size=$(get_file_size "$COMPRESSED_ARCHIVE")
    success "Created optimized archive: $COMPRESSED_ARCHIVE ($(format_bytes $archive_size))"
    
    # Clean up temp directory
    rm -rf "$TEMP_DIR"
}

show_optimization_report() {
    header "📊 OPTIMIZATION REPORT"
    
    local original_size=$(get_dir_size ".")
    original_size=$((original_size * 1024))
    
    echo "🗂️  Files deleted: $DELETED_FILES"
    echo "📁 Directories deleted: $DELETED_DIRS" 
    echo "📦 Files compressed: $COMPRESSED_FILES"
    echo "💾 Space saved: $(format_bytes $BYTES_SAVED)"
    echo ""
    
    if [ -f "$COMPRESSED_ARCHIVE" ]; then
        local archive_size=$(get_file_size "$COMPRESSED_ARCHIVE")
        local compression_ratio=$((archive_size * 100 / original_size))
        
        echo "📦 Original size: $(format_bytes $original_size)"
        echo "📦 Compressed size: $(format_bytes $archive_size)"
        echo "📊 Compression ratio: $compression_ratio%"
        echo ""
    fi
    
    echo "✅ Optimization completed successfully!"
    echo ""
    echo "📋 What was optimized:"
    echo "  • Removed Python cache files (__pycache__, .pyc)"
    echo "  • Cleaned Rust build artifacts (target/ directories)"
    echo "  • Removed system files (.DS_Store, Thumbs.db)"
    echo "  • Cleaned temporary files and logs"
    echo "  • Optimized notebook (removed outputs)"
    echo "  • Compressed Rust source code"
    echo "  • Created minimal requirements"
    echo "  • Generated optimized install script"
    echo ""
    echo "🚀 Ready for deployment:"
    echo "  • Extract: tar -xzf $COMPRESSED_ARCHIVE"
    echo "  • Install: ./install_optimized.sh"
    echo "  • Test: ./test.sh"
}

cleanup_optimization() {
    header "🧹 FINAL CLEANUP"
    
    # Remove temporary files created during optimization
    safe_remove "rust_modules_source.tar.gz" "temporary Rust archive"
    safe_remove "requirements_minimal.txt" "temporary requirements"
    safe_remove "install_optimized.sh" "temporary install script"
    safe_remove "$TEMP_DIR" "temporary directory"
    
    success "Optimization cleanup completed"
}

main() {
    clear
    header "🚀 CRYPTO TRADING SYSTEM OPTIMIZER"
    log "Starting repository optimization..."
    
    # Check if running from correct directory
    if [ ! -f "data_engine.py" ] || [ ! -f "neural_core.py" ]; then
        error "Please run this script from the trading system root directory"
        exit 1
    fi
    
    # Create backup first
    create_backup
    
    # Perform optimizations
    clean_python_cache
    clean_rust_artifacts
    clean_system_files
    clean_logs_temp
    
    # Optimize files
    optimize_python_files
    optimize_notebook
    compress_rust_source
    
    # Create deployment files
    create_minimal_requirements
    create_install_script
    create_optimized_archive
    
    # Show results
    show_optimization_report
    
    # Optional: Clean up temporary files
    read -p "🗑️  Remove temporary optimization files? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup_optimization
    fi
    
    success "Repository optimization completed!"
    log "Backup available in: $BACKUP_DIR"
    log "Optimized archive: $COMPRESSED_ARCHIVE"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
EOF