#!/usr/bin/env python3
import asyncio
import time
from datetime import datetime
from data_engine import RealTimeDataEngine
from execution_engine import HighFrequencyTradingEngine

async def main():
    print("🚀 LIVE PAPER TRADING SYSTEM")
    print("=" * 50)
    print(f"💰 Starting Capital: $1,000")
    print(f"🎯 Target: $10,000 (10x return)")
    print(f"⏰ Started: {datetime.now()}")
    print("=" * 50)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    
    # Initialize trading engine
    engine = HighFrequencyTradingEngine(symbols, initial_capital=1000.0)
    
    # Mock API credentials (for paper trading)
    mock_credentials = {
        'binance': {
            'api_key': 'paper_trading_key',
            'secret': 'paper_trading_secret',
            'sandbox': True
        }
    }
    
    await engine.initialize(mock_credentials)
    
    print("✅ All systems initialized")
    print("🧪 Paper trading mode - NO REAL MONEY AT RISK")
    print("\nStarting trading loop...")
    
    await engine.start_trading()

if __name__ == "__main__":
    asyncio.run(main())
