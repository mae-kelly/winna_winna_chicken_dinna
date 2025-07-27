#!/usr/bin/env python3

import asyncio
import sys
import os
import subprocess
import signal
import time
from pathlib import Path

def print_banner():
    print("🚀 HYPER-MOMENTUM CRYPTO TRADING SYSTEM")
    print("=" * 50)
    print("💰 Target: $1K → $10K (10x return)")
    print("⚡ GPU-Accelerated ML Pipeline")
    print("🔥 Real-time Token Breakout Detection")
    print("=" * 50)

def check_environment():
    """Check if environment is properly set up"""
    required_files = [
        'scanner.py',
        'executor.py', 
        'test_system.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def setup_environment():
    """Set up environment variables"""
    # Default to paper trading unless explicitly set
    if 'TRADING_MODE' not in os.environ:
        os.environ['TRADING_MODE'] = 'paper'
    
    # GPU optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

async def quick_test():
    """Run quick system test"""
    print("🧪 Running quick system test...")
    
    try:
        result = subprocess.run([
            sys.executable, 'test_system.py', 'test'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Quick test passed!")
            return True
        else:
            print("❌ Quick test failed!")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

async def run_scanner_only():
    """Run scanner only for testing signals"""
    print("🔍 Starting scanner in test mode...")
    print("Press Ctrl+C to stop")
    
    try:
        process = subprocess.Popen([
            sys.executable, 'scanner.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in iter(process.stdout.readline, ''):
            if line.strip():
                print(f"📊 Signal: {line.strip()}")
        
    except KeyboardInterrupt:
        print("\n🛑 Scanner stopped")
        process.terminate()

async def run_paper_trading():
    """Run full paper trading system"""
    print("📄 Starting PAPER TRADING mode...")
    print("💡 Using real market prices, simulated orders")
    print("Press Ctrl+C to stop")
    
    os.environ['TRADING_MODE'] = 'paper'
    
    try:
        # Start scanner
        scanner = subprocess.Popen([
            sys.executable, 'scanner.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        await asyncio.sleep(2)
        
        # Start executor with scanner output piped to it
        executor = subprocess.Popen([
            sys.executable, 'executor.py'
        ], stdin=scanner.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Monitor both processes
        while True:
            await asyncio.sleep(1)
            
            if scanner.poll() is not None:
                print("❌ Scanner process stopped")
                break
            
            if executor.poll() is not None:
                print("❌ Executor process stopped")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Paper trading stopped")
    finally:
        try:
            scanner.terminate()
            executor.terminate()
            scanner.wait(timeout=5)
            executor.wait(timeout=5)
        except:
            scanner.kill()
            executor.kill()

async def run_live_trading():
    """Run live trading with real money - EXTREME CAUTION"""
    print("🔴 LIVE TRADING MODE - REAL MONEY AT RISK!")
    print("⚠️  This will place real orders with real money")
    print("⚠️  You can lose money rapidly")
    print("⚠️  Only proceed if you understand the risks")
    
    # Multiple confirmations required
    confirm1 = input("\nType 'I UNDERSTAND THE RISKS' to continue: ")
    if confirm1 != "I UNDERSTAND THE RISKS":
        print("❌ Live trading cancelled")
        return
    
    print("\n🔴 API CREDENTIALS REQUIRED")
    print("Set these environment variables:")
    print("  export OKX_API_KEY='your_api_key'")
    print("  export OKX_SECRET_KEY='your_secret_key'") 
    print("  export OKX_PASSPHRASE='your_passphrase'")
    
    # Check API credentials
    required_vars = ['OKX_API_KEY', 'OKX_SECRET_KEY', 'OKX_PASSPHRASE']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"\n❌ Missing credentials: {', '.join(missing_vars)}")
        return
    
    confirm2 = input("\nType 'START LIVE TRADING' to proceed: ")
    if confirm2 != "START LIVE TRADING":
        print("❌ Live trading cancelled")
        return
    
    print("\n🔴 STARTING LIVE TRADING...")
    print("🔴 REAL MONEY AT RISK - Monitor closely!")
    
    os.environ['TRADING_MODE'] = 'live'
    
    try:
        # Start scanner
        scanner = subprocess.Popen([
            sys.executable, 'scanner.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        await asyncio.sleep(2)
        
        # Start executor
        executor = subprocess.Popen([
            sys.executable, 'executor.py'
        ], stdin=scanner.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Monitor processes
        while True:
            await asyncio.sleep(1)
            
            if scanner.poll() is not None:
                print("❌ Scanner process stopped")
                break
            
            if executor.poll() is not None:
                print("❌ Executor process stopped")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 LIVE TRADING STOPPED")
    finally:
        try:
            scanner.terminate()
            executor.terminate()
            scanner.wait(timeout=5)
            executor.wait(timeout=5)
        except:
            scanner.kill()
            executor.kill()

def show_menu():
    """Show main menu"""
    print("\n📋 TRADING SYSTEM MENU")
    print("=" * 30)
    print("1. Quick Test")
    print("2. Scanner Only (Test Signals)")
    print("3. Paper Trading (Recommended)")
    print("4. Live Trading (⚠️  REAL MONEY)")
    print("5. Full Test Suite")
    print("6. Monitor System")
    print("0. Exit")
    print("=" * 30)

async def main():
    print_banner()
    
    if not check_environment():
        print("❌ Environment check failed")
        return
    
    setup_environment()
    
    while True:
        show_menu()
        choice = input("\nSelect option (0-6): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        
        elif choice == '1':
            success = await quick_test()
            if success:
                print("✅ System ready for trading!")
            else:
                print("❌ Fix issues before trading")
        
        elif choice == '2':
            await run_scanner_only()
        
        elif choice == '3':
            await run_paper_trading()
        
        elif choice == '4':
            await run_live_trading()
        
        elif choice == '5':
            try:
                subprocess.run([sys.executable, 'test_system.py', 'test'])
            except KeyboardInterrupt:
                print("\n🛑 Test interrupted")
        
        elif choice == '6':
            try:
                subprocess.run([sys.executable, 'test_system.py', 'monitor'])
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped")
        
        else:
            print("❌ Invalid choice")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 System launcher stopped")
    except Exception as e:
        print(f"❌ Launcher error: {e}")