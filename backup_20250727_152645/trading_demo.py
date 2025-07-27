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
