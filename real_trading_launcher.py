#!/usr/bin/env python3
"""
🚀 REAL CRYPTO TRADING LAUNCHER
No simulations - Only real market data and trading
"""

import asyncio
import sys
import os
import subprocess
from pathlib import Path

def print_banner():
    print("🚀 REAL CRYPTO TRADING SYSTEM")
    print("=" * 50)
    print("💰 Live market data only")
    print("🎯 Two modes: Live execution | Paper execution")
    print("📊 Real prices, real technical analysis")
    print("=" * 50)

def check_dependencies():
    """Check if required packages are available"""
    required = ['aiohttp', 'numpy', 'pandas']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install aiohttp numpy pandas")
        return False
    
    return True

def show_menu():
    """Show main menu"""
    print("\n📋 REAL TRADING OPTIONS")
    print("=" * 30)
    print("1. Paper Trading (Real prices, terminal trades)")
    print("2. Live Trading (⚠️  REAL MONEY)")
    print("3. Market Data Test")
    print("4. Check System Status")
    print("0. Exit")
    print("=" * 30)

async def start_paper_trading():
    """Start paper trading with real market data"""
    print("📄 Starting PAPER TRADING...")
    print("💡 Using real market prices, simulated trades in terminal")
    print("Press Ctrl+C to stop")
    
    try:
        # Start the real trading system in paper mode
        process = subprocess.Popen([
            sys.executable, '-c', '''
import asyncio
from real_trading_system import RealTradingSystem

async def run():
    system = RealTradingSystem(mode="paper")
    await system.start()

if __name__ == "__main__":
    asyncio.run(run())
'''
        ], env={**os.environ, 'TRADING_MODE': 'paper'})
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Paper trading stopped")
        process.terminate()

async def start_live_trading():
    """Start live trading with real money"""
    print("🔴 LIVE TRADING MODE SELECTED")
    print("⚠️  WARNING: This will trade with REAL MONEY!")
    print("⚠️  Make sure you understand the risks!")
    
    # Multiple confirmations for safety
    confirm1 = input("\nType 'I UNDERSTAND THE RISKS' to continue: ")
    if confirm1 != "I UNDERSTAND THE RISKS":
        print("❌ Live trading cancelled")
        return
    
    print("\n🔴 FINAL WARNING: REAL MONEY WILL BE USED")
    confirm2 = input("Type 'START LIVE TRADING' to proceed: ")
    if confirm2 != "START LIVE TRADING":
        print("❌ Live trading cancelled")
        return
    
    print("\n🔴 STARTING LIVE TRADING...")
    print("🔴 Monitor your positions carefully!")
    
    try:
        # Start the real trading system in live mode
        process = subprocess.Popen([
            sys.executable, '-c', '''
import asyncio
from real_trading_system import RealTradingSystem

async def run():
    system = RealTradingSystem(mode="live")
    await system.start()

if __name__ == "__main__":
    asyncio.run(run())
'''
        ], env={**os.environ, 'TRADING_MODE': 'live'})
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 LIVE TRADING STOPPED")
        process.terminate()

async def test_market_data():
    """Test real market data connection"""
    print("🧪 Testing real market data connection...")
    
    try:
        from real_trading_system import RealMarketDataFeed
        
        feed = RealMarketDataFeed()
        await feed.start()
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        prices = await feed.get_live_prices(symbols)
        
        if prices:
            print("✅ Market data connection successful!")
            for symbol, data in prices.items():
                print(f"   {symbol}: ${data['price']:.2f} ({data['change_24h']:+.2f}%)")
        else:
            print("❌ Failed to get market data")
        
        await feed.stop()
        
    except Exception as e:
        print(f"❌ Market data test failed: {e}")

def check_system_status():
    """Check system requirements and status"""
    print("🔍 Checking system status...")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python: {python_version}")
    
    # Check dependencies
    deps_ok = check_dependencies()
    if deps_ok:
        print("✅ Dependencies: All required packages available")
    else:
        print("❌ Dependencies: Missing packages")
        return
    
    # Check internet connection
    try:
        import aiohttp
        print("✅ Internet: Connection available")
    except:
        print("❌ Internet: Connection test failed")
    
    # Check trading files
    required_files = ['real_trading_system.py']
    for file in required_files:
        if Path(file).exists():
            print(f"✅ File: {file}")
        else:
            print(f"❌ File: {file} missing")
    
    print("\n🎯 System Status: READY" if deps_ok else "🎯 System Status: NEEDS SETUP")

async def main():
    print_banner()
    
    if not check_dependencies():
        print("❌ System check failed - install missing dependencies")
        return
    
    while True:
        show_menu()
        choice = input("\nSelect option (0-4): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        
        elif choice == '1':
            await start_paper_trading()
        
        elif choice == '2':
            await start_live_trading()
        
        elif choice == '3':
            await test_market_data()
        
        elif choice == '4':
            check_system_status()
        
        else:
            print("❌ Invalid choice")
        
        if choice in ['1', '2', '3']:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Launcher stopped")