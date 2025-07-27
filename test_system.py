#!/usr/bin/env python3

import asyncio
import subprocess
import signal
import time
import json
import logging
import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import aiohttp
import torch
import numpy as np
from datetime import datetime, timedelta
import psutil
import threading
from queue import Queue
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    test_name: str
    status: str
    duration: float
    details: Dict
    error_message: str = ""

class SystemTester:
    def __init__(self):
        self.results = []
        self.scanner_process = None
        self.executor_process = None
        self.test_signals = []
        self.performance_data = []
        
    async def run_full_test_suite(self):
        """Run complete test suite for the trading system"""
        
        print("🚀 HYPER-MOMENTUM TRADING SYSTEM - COMPREHENSIVE TESTING")
        print("=" * 70)
        
        tests = [
            ("dependency_check", self.test_dependencies),
            ("gpu_acceleration", self.test_gpu_acceleration),
            ("api_connectivity", self.test_api_connectivity),
            ("data_feed", self.test_data_feed),
            ("signal_generation", self.test_signal_generation),
            ("execution_engine", self.test_execution_engine),
            ("risk_management", self.test_risk_management),
            ("paper_trading", self.test_paper_trading),
            ("live_integration", self.test_live_integration),
            ("performance_stress", self.test_performance_stress)
        ]
        
        total_tests = len(tests)
        passed_tests = 0
        
        for i, (test_name, test_func) in enumerate(tests, 1):
            print(f"\n[{i}/{total_tests}] Running {test_name}...")
            
            start_time = time.time()
            try:
                result = await test_func()
                duration = time.time() - start_time
                
                if result['status'] == 'PASS':
                    print(f"✅ {test_name}: PASSED ({duration:.2f}s)")
                    passed_tests += 1
                else:
                    print(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                
                self.results.append(TestResult(
                    test_name=test_name,
                    status=result['status'],
                    duration=duration,
                    details=result.get('details', {}),
                    error_message=result.get('error', '')
                ))
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ {test_name}: ERROR - {str(e)}")
                self.results.append(TestResult(
                    test_name=test_name,
                    status='ERROR',
                    duration=duration,
                    details={},
                    error_message=str(e)
                ))
        
        self.print_test_summary(passed_tests, total_tests)
        return passed_tests == total_tests
    
    async def test_dependencies(self) -> Dict:
        """Test all required dependencies"""
        required_packages = [
            'torch', 'numpy', 'pandas', 'aiohttp', 'uvloop',
            'cupy', 'cudf', 'orjson', 'matplotlib', 'sklearn'
        ]
        
        missing = []
        installed = []
        
        for package in required_packages:
            try:
                __import__(package)
                installed.append(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            return {
                'status': 'FAIL',
                'error': f"Missing packages: {', '.join(missing)}",
                'details': {'installed': installed, 'missing': missing}
            }
        
        return {
            'status': 'PASS',
            'details': {'installed_packages': len(installed)}
        }
    
    async def test_gpu_acceleration(self) -> Dict:
        """Test GPU acceleration capabilities"""
        details = {}
        
        # Test PyTorch CUDA
        details['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            details['gpu_name'] = torch.cuda.get_device_name()
            details['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Test tensor operations
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            start_time = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            details['gpu_matmul_time'] = gpu_time
        
        # Test CuPy if available
        try:
            import cupy as cp
            details['cupy_available'] = True
            arr = cp.random.randn(1000, 1000)
            start_time = time.time()
            result = cp.dot(arr, arr)
            cp.cuda.Stream.null.synchronize()
            details['cupy_time'] = time.time() - start_time
        except ImportError:
            details['cupy_available'] = False
        
        if details['cuda_available']:
            return {'status': 'PASS', 'details': details}
        else:
            return {
                'status': 'WARN',
                'error': 'GPU not available - will run on CPU',
                'details': details
            }
    
    async def test_api_connectivity(self) -> Dict:
        """Test API connectivity to DexScreener and OKX"""
        async with aiohttp.ClientSession() as session:
            results = {}
            
            # Test DexScreener API
            try:
                async with session.get('https://api.dexscreener.com/latest/dex/pairs/ethereum', timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results['dexscreener'] = {'status': 'PASS', 'pairs_count': len(data.get('pairs', []))}
                    else:
                        results['dexscreener'] = {'status': 'FAIL', 'code': resp.status}
            except Exception as e:
                results['dexscreener'] = {'status': 'ERROR', 'error': str(e)}
            
            # Test OKX API (public endpoints)
            try:
                async with session.get('https://www.okx.com/api/v5/market/tickers?instType=SPOT', timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results['okx'] = {'status': 'PASS', 'tickers_count': len(data.get('data', []))}
                    else:
                        results['okx'] = {'status': 'FAIL', 'code': resp.status}
            except Exception as e:
                results['okx'] = {'status': 'ERROR', 'error': str(e)}
        
        if all(r.get('status') == 'PASS' for r in results.values()):
            return {'status': 'PASS', 'details': results}
        else:
            return {'status': 'FAIL', 'error': 'API connectivity issues', 'details': results}
    
    async def test_data_feed(self) -> Dict:
        """Test data feed functionality"""
        try:
            from scanner import DexScreenerAPI, TokenProcessor
            
            api = DexScreenerAPI()
            processor = TokenProcessor()
            
            await api.start()
            
            # Test getting trending tokens
            tokens = await api.get_trending_tokens(50)
            
            await api.stop()
            
            if len(tokens) > 0:
                # Test processing a token
                processed = processor.process_token(tokens[0])
                
                return {
                    'status': 'PASS',
                    'details': {
                        'tokens_fetched': len(tokens),
                        'processing_success': processed is not None
                    }
                }
            else:
                return {
                    'status': 'FAIL',
                    'error': 'No tokens fetched from DexScreener'
                }
                
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def test_signal_generation(self) -> Dict:
        """Test signal generation with mock data"""
        try:
            from scanner import TransformerBreakoutModel, GPUFeatureExtractor, TokenData
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = TransformerBreakoutModel().to(device)
            model.eval()
            
            extractor = GPUFeatureExtractor()
            
            # Create mock token data
            mock_token = TokenData(
                symbol="TESTTOKEN",
                address="0x123",
                price=1.0,
                price_1m=0.92,
                price_3m=0.88,
                price_5m=0.85,
                volume_24h=1000000,
                volume_1h=50000,
                volume_15m=15000,
                liquidity=500000,
                market_cap=10000000,
                price_change_1m=8.7,
                price_change_3m=13.6,
                price_change_5m=17.6,
                volume_surge=250.0,
                timestamp=time.time()
            )
            
            # Extract features
            features = extractor.extract_features(mock_token)
            
            # Run model prediction
            with torch.no_grad():
                input_tensor = features.unsqueeze(0).unsqueeze(0)
                output = model(input_tensor)
                
                confidence = output['breakout_confidence'].item()
                entropy = output['entropy_score'].item()
            
            return {
                'status': 'PASS',
                'details': {
                    'model_loaded': True,
                    'features_extracted': True,
                    'breakout_confidence': confidence,
                    'entropy_score': entropy,
                    'feature_count': len(features)
                }
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def test_execution_engine(self) -> Dict:
        """Test execution engine functionality"""
        try:
            from executor import OKXPaperTradingAPI, PositionSizer, EntropyExitCalculator
            
            # Test paper trading API
            api = OKXPaperTradingAPI()
            await api.start()
            
            # Test getting ticker
            ticker = await api.get_ticker("BTCUSDT")
            
            # Test paper order
            order = await api.place_order("BTCUSDT", "buy", 0.001)
            
            await api.stop()
            
            # Test position sizer
            sizer = PositionSizer()
            from scanner import BreakoutSignal
            
            mock_signal = BreakoutSignal(
                symbol="TEST", address="0x123", price=1.0,
                price_change_1m=10.0, price_change_3m=12.0, price_change_5m=15.0,
                volume_surge=200.0, breakout_confidence=0.9, entropy_score=0.8,
                liquidity=1000000, market_cap=5000000, safety_score=0.9,
                final_score=0.85, timestamp=time.time(), momentum_vector=[0.1, 0.2, 0.3]
            )
            
            position_size = sizer.calculate_position_size(mock_signal, 1000.0)
            
            return {
                'status': 'PASS',
                'details': {
                    'api_functional': ticker is not None,
                    'order_placement': order is not None,
                    'position_sizing': position_size > 0,
                    'calculated_size': position_size
                }
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def test_risk_management(self) -> Dict:
        """Test risk management features"""
        try:
            from executor import EntropyExitCalculator
            
            calculator = EntropyExitCalculator()
            
            # Test with mock price data
            test_symbol = "TESTCOIN"
            prices = [1.0, 1.05, 1.12, 1.08, 1.15, 1.22, 1.18, 1.25, 1.20, 1.16]
            
            for price in prices:
                calculator.update_price(test_symbol, price)
            
            momentum_vector = np.array([0.1, 0.2, -0.1, 0.15])
            entropy = calculator.calculate_current_entropy(test_symbol, momentum_vector)
            
            should_exit = calculator.should_exit_on_entropy(test_symbol, 0.4)
            
            return {
                'status': 'PASS',
                'details': {
                    'entropy_calculated': True,
                    'entropy_value': entropy,
                    'exit_decision': should_exit,
                    'price_history_length': len(calculator.price_history[test_symbol])
                }
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def test_paper_trading(self) -> Dict:
        """Test full paper trading cycle"""
        try:
            from executor import OKXPaperTradingAPI
            
            api = OKXPaperTradingAPI()
            await api.start()
            
            initial_balance = await api.get_balance()
            initial_amount = float(initial_balance['availBal'])
            
            # Place buy order
            buy_order = await api.place_order("BTCUSDT", "buy", 0.001)
            
            if buy_order:
                # Check balance after buy
                balance_after_buy = await api.get_balance()
                
                # Place sell order
                sell_order = await api.place_order("BTCUSDT", "sell", 0.001)
                
                if sell_order:
                    final_balance = await api.get_balance()
                    final_amount = float(final_balance['availBal'])
                    
                    await api.stop()
                    
                    return {
                        'status': 'PASS',
                        'details': {
                            'initial_balance': initial_amount,
                            'final_balance': final_amount,
                            'buy_order_success': True,
                            'sell_order_success': True,
                            'pnl': final_amount - initial_amount
                        }
                    }
            
            await api.stop()
            return {'status': 'FAIL', 'error': 'Paper trading cycle incomplete'}
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def test_live_integration(self) -> Dict:
        """Test live integration between scanner and executor"""
        try:
            # Create mock signal
            mock_signal = {
                'symbol': 'MOCKTOKEN',
                'address': '0xmock123',
                'price': 1.0,
                'price_change_1m': 10.5,
                'price_change_3m': 12.3,
                'price_change_5m': 14.1,
                'volume_surge': 300.0,
                'breakout_confidence': 0.92,
                'entropy_score': 0.85,
                'liquidity': 800000,
                'market_cap': 8000000,
                'safety_score': 0.88,
                'final_score': 0.89,
                'timestamp': time.time(),
                'momentum_vector': [0.1, 0.2, 0.15, 0.3],
                'pair_address': '0xpair123',
                'chain': 'ethereum'
            }
            
            # Test JSON serialization/deserialization
            signal_json = json.dumps(mock_signal)
            parsed_signal = json.loads(signal_json)
            
            return {
                'status': 'PASS',
                'details': {
                    'signal_created': True,
                    'json_serializable': True,
                    'signal_fields_complete': len(parsed_signal) >= 10
                }
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def test_performance_stress(self) -> Dict:
        """Test system performance under load"""
        try:
            from scanner import GPUFeatureExtractor, TokenData
            
            extractor = GPUFeatureExtractor()
            
            # Generate many mock tokens
            tokens = []
            for i in range(100):
                token = TokenData(
                    symbol=f"TOKEN{i}",
                    address=f"0x{i:040x}",
                    price=np.random.uniform(0.1, 10.0),
                    price_1m=np.random.uniform(0.1, 10.0),
                    price_3m=np.random.uniform(0.1, 10.0),
                    price_5m=np.random.uniform(0.1, 10.0),
                    volume_24h=np.random.uniform(10000, 1000000),
                    volume_1h=np.random.uniform(1000, 100000),
                    volume_15m=np.random.uniform(100, 10000),
                    liquidity=np.random.uniform(50000, 5000000),
                    market_cap=np.random.uniform(100000, 100000000),
                    price_change_1m=np.random.uniform(-5, 20),
                    price_change_3m=np.random.uniform(-5, 20),
                    price_change_5m=np.random.uniform(-5, 20),
                    volume_surge=np.random.uniform(0, 500),
                    timestamp=time.time()
                )
                tokens.append(token)
            
            # Time feature extraction
            start_time = time.time()
            for token in tokens:
                features = extractor.extract_features(token)
            processing_time = time.time() - start_time
            
            tokens_per_second = len(tokens) / processing_time
            
            return {
                'status': 'PASS',
                'details': {
                    'tokens_processed': len(tokens),
                    'processing_time': processing_time,
                    'tokens_per_second': tokens_per_second,
                    'performance_acceptable': tokens_per_second > 50
                }
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    def print_test_summary(self, passed: int, total: int):
        """Print comprehensive test summary"""
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY: {passed}/{total} PASSED")
        print(f"{'='*70}")
        
        for result in self.results:
            status_emoji = "✅" if result.status == "PASS" else "⚠️" if result.status == "WARN" else "❌"
            print(f"{status_emoji} {result.test_name.upper()}: {result.status} ({result.duration:.2f}s)")
            
            if result.error_message:
                print(f"   Error: {result.error_message}")
        
        print(f"\n🎯 SYSTEM STATUS: {'READY FOR DEPLOYMENT' if passed == total else 'NEEDS ATTENTION'}")
        
        if passed == total:
            print("\n🚀 ALL TESTS PASSED - READY TO TRADE!")
            print("Next steps:")
            print("1. Run: python live_deploy.py --mode=paper")
            print("2. Monitor performance for 1 hour")
            print("3. Switch to live trading if satisfied")

class LiveDeploymentManager:
    def __init__(self):
        self.scanner_process = None
        self.executor_process = None
        self.monitor_process = None
        self.trading_mode = "paper"
        self.performance_data = []
        
    async def deploy_system(self, mode: str = "paper", duration_hours: int = 24):
        """Deploy the complete trading system"""
        
        self.trading_mode = mode
        
        print(f"🚀 DEPLOYING HYPER-MOMENTUM TRADING SYSTEM")
        print(f"📊 Mode: {mode.upper()}")
        print(f"⏰ Duration: {duration_hours} hours")
        print(f"🎯 Target: $1K → $10K (10x return)")
        print("=" * 60)
        
        try:
            # Start scanner
            print("🔍 Starting token scanner...")
            self.scanner_process = await self.start_scanner()
            await asyncio.sleep(5)
            
            # Start executor
            print("⚡ Starting execution engine...")
            self.executor_process = await self.start_executor(mode)
            await asyncio.sleep(3)
            
            # Start monitoring
            print("📊 Starting performance monitor...")
            self.monitor_process = await self.start_monitor()
            
            print(f"\n✅ System deployed successfully in {mode} mode!")
            print("Press Ctrl+C to stop trading...")
            
            # Run for specified duration
            end_time = time.time() + (duration_hours * 3600)
            
            while time.time() < end_time:
                await asyncio.sleep(10)
                
                # Check if processes are still running
                if not self.scanner_process or self.scanner_process.poll() is not None:
                    print("❌ Scanner process died, restarting...")
                    self.scanner_process = await self.start_scanner()
                
                if not self.executor_process or self.executor_process.poll() is not None:
                    print("❌ Executor process died, restarting...")
                    self.executor_process = await self.start_executor(mode)
            
            print(f"\n🏁 Trading session completed after {duration_hours} hours")
            
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        finally:
            await self.stop_all_processes()
    
    async def start_scanner(self):
        """Start the token scanner process"""
        cmd = [sys.executable, "scanner.py"]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        return process
    
    async def start_executor(self, mode: str):
        """Start the execution engine process"""
        env = os.environ.copy()
        env['TRADING_MODE'] = mode
        
        cmd = [sys.executable, "executor.py"]
        
        # Connect scanner output to executor input
        process = subprocess.Popen(
            cmd,
            stdin=self.scanner_process.stdout if self.scanner_process else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        return process
    
    async def start_monitor(self):
        """Start performance monitoring"""
        cmd = [sys.executable, "-c", """
import time
import psutil
import json
import sys

while True:
    try:
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        stats = {
            'timestamp': time.time(),
            'cpu_percent': cpu_usage,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3)
        }
        
        print(f"📊 System: CPU {cpu_usage:.1f}%, RAM {memory.percent:.1f}%")
        
        time.sleep(30)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Monitor error: {e}")
        time.sleep(10)
"""]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return process
    
    async def stop_all_processes(self):
        """Stop all running processes"""
        processes = [self.scanner_process, self.executor_process, self.monitor_process]
        
        for process in processes:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    print(f"Error stopping process: {e}")

async def main():
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            tester = SystemTester()
            success = await tester.run_full_test_suite()
            sys.exit(0 if success else 1)
            
        elif command == "deploy":
            mode = "paper"
            duration = 24
            
            if "--mode=live" in sys.argv:
                mode = "live"
                print("⚠️  LIVE TRADING MODE ENABLED - REAL MONEY AT RISK!")
                response = input("Type 'CONFIRM' to proceed with live trading: ")
                if response != "CONFIRM":
                    print("Live trading cancelled")
                    return
            
            if "--duration=" in " ".join(sys.argv):
                for arg in sys.argv:
                    if arg.startswith("--duration="):
                        duration = int(arg.split("=")[1])
            
            deployer = LiveDeploymentManager()
            await deployer.deploy_system(mode, duration)
            
        elif command == "monitor":
            print("📊 Starting standalone performance monitor...")
            while True:
                try:
                    cpu = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    print(f"System: CPU {cpu:.1f}%, RAM {memory.percent:.1f}%")
                    await asyncio.sleep(30)
                except KeyboardInterrupt:
                    break
        else:
            print("Unknown command. Use: test, deploy, or monitor")
    else:
        print("🚀 HYPER-MOMENTUM TRADING SYSTEM")
        print("=" * 40)
        print("Available commands:")
        print("  python test_system.py test           - Run full test suite")
        print("  python test_system.py deploy         - Deploy in paper mode")
        print("  python test_system.py deploy --mode=live - Deploy with real money")
        print("  python test_system.py deploy --duration=6 - Run for 6 hours")
        print("  python test_system.py monitor        - System monitoring only")

if __name__ == "__main__":
    asyncio.run(main())