# 🚀 Real Crypto Trading System

**Live market data only - No simulations or fake data**

## 🎯 Two Trading Modes

### 1. Paper Trading Mode
- ✅ **Real live prices** from OKX exchange
- ✅ **Real technical analysis** (RSI, Moving Averages)
- ✅ **Simulated trades** shown in terminal
- ✅ **Safe testing** - no real money at risk

### 2. Live Trading Mode
- 🔴 **Real live prices** from OKX exchange
- 🔴 **Real technical analysis** 
- 🔴 **Real trades** with real money
- ⚠️ **HIGH RISK** - only for experienced traders

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install aiohttp numpy pandas matplotlib
```

### 2. Start Trading
```bash
# Easy launcher
python real_trading_launcher.py

# Or direct paper trading
python real_trading_system.py --paper

# Or direct live trading (⚠️ REAL MONEY)
python real_trading_system.py --live
```

## 📊 Features

### Real Market Data
- Live price feeds from OKX exchange
- Real-time RSI calculation
- Moving average analysis (20/50 period)
- 24/7 market monitoring

### Trading Strategy
- **Buy Signal**: RSI < 30 + uptrend + price above MA20
- **Sell Signal**: RSI > 70 + downtrend + price below MA20
- **Stop Loss**: 5% below entry price
- **Take Profit**: 10% above entry price

### Risk Management
- Maximum 10% of capital per trade
- 5-minute cooldown between signals
- Automatic stop loss / take profit
- Position monitoring

## 🛡️ Safety Features

### Paper Trading Safety
- ✅ Real market prices
- ✅ Real technical indicators
- ✅ Simulated balance tracking
- ✅ Terminal-only trade execution
- ✅ No API keys required
- ✅ Zero financial risk

### Live Trading Protection
- 🔴 Multiple confirmation prompts
- 🔴 Conservative position sizing
- 🔴 Automatic risk controls
- 🔴 Stop loss protection
- 🔴 Real-time monitoring

## 📈 Supported Symbols

- **BTCUSDT** - Bitcoin
- **ETHUSDT** - Ethereum  
- **BNBUSDT** - Binance Coin
- **SOLUSDT** - Solana

## 🔧 Configuration

### Environment Variables
```bash
# Set trading mode
export TRADING_MODE=paper  # or "live"
```

### Exchange API (Live Trading Only)
For live trading, you'll need to add your exchange API credentials to the code.

## 📊 Example Output

### Paper Trading
```
📈 PAPER BUY: 0.001500 BTCUSDT @ $67,234.50
💰 Balance: $899.12

🎯 TRADING SIGNAL:
   Symbol: ETHUSDT
   Signal: BUY
   Confidence: 78%
   Price: $3,456.78

📉 PAPER SELL: 0.289234 ETHUSDT @ $3,523.45
💰 P&L: $19.28 | Balance: $918.40
```

### Live Trading
```
🔴 LIVE TRADE: BUY 0.001500 BTCUSDT @ $67,234.50
🔴 THIS WOULD PLACE A REAL ORDER WITH REAL MONEY!
```

## ⚠️ Important Warnings

### Paper Trading
- Uses real market data and prices
- No financial risk
- Perfect for learning and testing
- Terminal output only

### Live Trading
- **REAL MONEY AT RISK**
- Can result in significant losses
- Only use money you can afford to lose
- Requires exchange API setup
- Monitor positions constantly

## 🎯 Performance Monitoring

The system provides real-time monitoring:
- Current balance and P&L
- Active positions
- Trading signals generated
- Win rate and performance metrics

## 🚨 Disclaimer

This is a trading system that can result in financial losses. 

**Paper Trading**: Safe for learning and testing strategies.

**Live Trading**: High risk - only use with money you can afford to lose completely.

Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors.

## 📞 Getting Help

1. Start with paper trading mode
2. Monitor performance for several hours
3. Understand the strategy before considering live trading
4. Never risk more than you can afford to lose

---

## 🏁 Quick Commands

```bash
# Paper trading (recommended start)
python real_trading_launcher.py

# Quick market data test
python real_trading_system.py --test

# Check system status
python -c "from real_trading_launcher import check_system_status; check_system_status()"
```

**Remember**: Start with paper trading to learn the system before risking real money!