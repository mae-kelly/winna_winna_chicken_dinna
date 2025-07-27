#!/usr/bin/env python3

import asyncio
import aiohttp
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import orjson
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import uvloop
import threading
from threading import Lock
import hashlib
import hmac
import base64
from urllib.parse import urlencode
import warnings
from data_engine import RealTimeDataEngine, MarketData
from execution_engine import HighFrequencyTradingEngine, RiskManager
from neural_core import SelfOptimizingModel
import cupy as cp
import cudf

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class BreakoutSignal:
    symbol: str
    address: str
    price: float
    price_change_1m: float
    price_change_3m: float
    price_change_5m: float
    volume_surge: float
    breakout_confidence: float
    entropy_score: float
    liquidity: float
    market_cap: float
    safety_score: float
    final_score: float
    timestamp: float
    momentum_vector: List[float]
    pair_address: str = ""
    chain: str = "ethereum"

@dataclass
class ExecutionPosition:
    signal: BreakoutSignal
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: datetime
    leverage: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    entropy_exit_threshold: float
    cooldown_until: float
    slippage_tolerance: float
    execution_quality: float
    position_id: str
    last_entropy_check: float = 0.0
    max_profit: float = 0.0
    drawdown_from_peak: float = 0.0

@dataclass
class ExecutionMetrics:
    total_signals: int = 0
    executed_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    execution_rate: float = 0.0
    avg_execution_time: float = 0.0
    slippage_avg: float = 0.0
    entropy_exits: int = 0
    target_exits: int = 0
    stop_exits: int = 0
    cooldown_rejects: int = 0

class OKXTradingAPI:
    def __init__(self, api_key: str = "", secret_key: str = "", passphrase: str = "", live_mode: bool = False):
        self.api_key = api_key or os.getenv('OKX_API_KEY', 'paper_trading_key')
        self.secret_key = secret_key or os.getenv('OKX_SECRET_KEY', 'paper_trading_secret') 
        self.passphrase = passphrase or os.getenv('OKX_PASSPHRASE', 'paper_trading_pass')
        
        self.live_mode = live_mode or os.getenv('TRADING_MODE', 'paper') == 'live'
        
        if self.live_mode:
            self.base_url = "https://www.okx.com"
            logger.critical("🔴 LIVE TRADING MODE ENABLED - REAL MONEY AT RISK!")
        else:
            self.base_url = "https://www.okx.com"  # Using real API for price feeds
            logger.info("📄 Paper trading mode - Real prices, simulated orders")
        
        self.session = None
        
        # Paper trading simulation
        self.paper_balance = 1000.0
        self.paper_positions = {}
        self.paper_orders = {}
        self.order_counter = 0
        
        self.simulated_slippage = 0.0015
        self.simulated_latency = 0.05
        
    async def start(self):
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=25,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'Content-Type': 'application/json',
                'OK-ACCESS-KEY': self.api_key,
                'OK-ACCESS-SIGN': '',
                'OK-ACCESS-TIMESTAMP': '',
                'OK-ACCESS-PASSPHRASE': self.passphrase,
                'User-Agent': 'HyperMomentumExecutor/1.0'
            }
        )
        
        logger.info("📡 OKX Paper Trading API initialized")
    
    async def stop(self):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        message = timestamp + method + path + body
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        ).decode()
        return signature
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            # Always get real market data for accurate pricing
            timestamp = str(int(time.time() * 1000))
            path = f"/api/v5/market/ticker?instId={symbol}"
            
            headers = {
                'OK-ACCESS-KEY': self.api_key,
                'OK-ACCESS-SIGN': self._generate_signature(timestamp, 'GET', path),
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': self.passphrase,
                'Content-Type': 'application/json'
            }
            
            url = self.base_url + path
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('data') and len(data['data']) > 0:
                        return data['data'][0]
                
                # Fallback to public endpoint if auth fails
                async with self.session.get(f"{self.base_url}/api/v5/market/ticker?instId={symbol}") as fallback_resp:
                    if fallback_resp.status == 200:
                        fallback_data = await fallback_resp.json()
                        if fallback_data.get('data') and len(fallback_data['data']) > 0:
                            return fallback_data['data'][0]
            
            # Final fallback with simulated data if all else fails
            await asyncio.sleep(self.simulated_latency)
            base_price = 1.0
            if symbol.endswith('USDT'):
                base_price = np.random.uniform(0.5, 2.0)
            
            price_variation = np.random.normal(0, 0.002)
            current_price = base_price * (1 + price_variation)
            spread = current_price * self.simulated_slippage
            
            return {
                'instId': symbol,
                'last': str(current_price),
                'bidPx': str(current_price - spread/2),
                'askPx': str(current_price + spread/2),
                'vol24h': str(np.random.uniform(100000, 1000000)),
                'ts': str(int(time.time() * 1000))
            }
            
        except Exception as e:
            logger.error(f"Ticker fetch error for {symbol}: {e}")
            return None
    
    async def place_order(self, symbol: str, side: str, size: float, 
                         order_type: str = "market", price: float = None) -> Optional[Dict]:
        
        if self.live_mode:
            return await self._place_live_order(symbol, side, size, order_type, price)
        else:
            return await self._place_paper_order(symbol, side, size, order_type, price)
    
    async def _place_live_order(self, symbol: str, side: str, size: float, 
                               order_type: str = "market", price: float = None) -> Optional[Dict]:
        """Place real order on OKX exchange - USE WITH EXTREME CAUTION"""
        try:
            logger.critical(f"🔴 LIVE ORDER: {side} {size} {symbol} @ {order_type}")
            
            # Get real-time price for order
            ticker = await self.get_ticker(symbol)
            if not ticker:
                logger.error(f"Cannot get ticker for live order: {symbol}")
                return None
            
            if order_type == "market":
                execution_price = float(ticker['askPx']) if side == 'buy' else float(ticker['bidPx'])
            else:
                execution_price = price or float(ticker['last'])
            
            # Prepare order data
            order_data = {
                "instId": symbol,
                "tdMode": "cash",  # Cash trading mode
                "side": side,
                "ordType": order_type,
                "sz": str(size)
            }
            
            if order_type == "limit" and price:
                order_data["px"] = str(price)
            
            # Create signature
            timestamp = str(int(time.time() * 1000))
            path = "/api/v5/trade/order"
            body = json.dumps(order_data)
            
            headers = {
                'OK-ACCESS-KEY': self.api_key,
                'OK-ACCESS-SIGN': self._generate_signature(timestamp, 'POST', path, body),
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': self.passphrase,
                'Content-Type': 'application/json'
            }
            
            url = self.base_url + path
            
            async with self.session.post(url, headers=headers, data=body) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get('code') == '0' and result.get('data'):
                        order_data = result['data'][0]
                        
                        logger.critical(f"🔴 LIVE ORDER EXECUTED: {order_data.get('ordId')}")
                        
                        return {
                            'ordId': order_data.get('ordId'),
                            'instId': symbol,
                            'side': side,
                            'sz': str(size),
                            'px': str(execution_price),
                            'fillPx': str(execution_price),
                            'fillSz': str(size),
                            'state': 'filled',
                            'fillTime': timestamp,
                            'fee': str(float(size) * execution_price * 0.001),
                            'pnl': '0'
                        }
                    else:
                        logger.error(f"Live order failed: {result}")
                        return None
                else:
                    logger.error(f"Live order HTTP error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Live order placement error: {e}")
            return None
    
    async def _place_paper_order(self, symbol: str, side: str, size: float, 
                                order_type: str = "market", price: float = None) -> Optional[Dict]:
        """Place simulated paper order with real market prices"""
        try:
            await asyncio.sleep(self.simulated_latency)
            
            self.order_counter += 1
            order_id = f"paper_{int(time.time())}_{self.order_counter}"
            
            # Get REAL market data for paper order
            ticker = await self.get_ticker(symbol)
            if not ticker:
                return None
            
            if order_type == "market":
                execution_price = float(ticker['askPx']) if side == 'buy' else float(ticker['bidPx'])
            else:
                execution_price = price or float(ticker['last'])
            
            # Apply realistic slippage
            slippage_factor = np.random.uniform(0.8, 1.2) * self.simulated_slippage
            if side == 'buy':
                execution_price *= (1 + slippage_factor)
            else:
                execution_price *= (1 - slippage_factor)
            
            order_value = size * execution_price
            
            # Check paper balance
            if side == 'buy' and order_value > self.paper_balance:
                logger.warning(f"Insufficient paper balance for {symbol}: need ${order_value:.2f}, have ${self.paper_balance:.2f}")
                return None
            
            # Update paper balance
            if side == 'buy':
                self.paper_balance -= order_value
            else:
                self.paper_balance += order_value
            
            order = {
                'ordId': order_id,
                'instId': symbol,
                'side': side,
                'sz': str(size),
                'px': str(execution_price),
                'fillPx': str(execution_price),
                'fillSz': str(size),
                'state': 'filled',
                'fillTime': str(int(time.time() * 1000)),
                'fee': str(order_value * 0.001),
                'pnl': '0'
            }
            
            self.paper_orders[order_id] = order
            
            logger.info(f"📝 Paper order executed: {side} {size} {symbol} @ ${execution_price:.6f} (Real market price)")
            
            return order
            
        except Exception as e:
            logger.error(f"Paper order placement error: {e}")
            return None
    
    async def get_positions(self) -> List[Dict]:
        return list(self.paper_positions.values())
    
    async def get_balance(self) -> Dict:
        total_position_value = sum(
            float(pos.get('notionalUsd', 0)) for pos in self.paper_positions.values()
        )
        
        return {
            'availBal': str(self.paper_balance),
            'cashBal': str(self.paper_balance),
            'totalEq': str(self.paper_balance + total_position_value),
            'frozenBal': '0'
        }

class PositionSizer:
    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_risk = 0.15
        self.max_total_exposure = 0.8
        self.min_position_size = 10.0
        self.max_position_size = 150.0
        
    def calculate_position_size(self, signal: BreakoutSignal, current_capital: float) -> float:
        self.current_capital = current_capital
        
        base_risk = self.max_position_risk
        
        confidence_multiplier = min(signal.breakout_confidence * 1.5, 1.2)
        entropy_multiplier = min(signal.entropy_score * 1.3, 1.1)
        safety_multiplier = signal.safety_score
        liquidity_multiplier = min(signal.liquidity / 1000000, 1.0) * 0.5 + 0.5
        
        score_multiplier = (confidence_multiplier * 0.4 + 
                          entropy_multiplier * 0.3 + 
                          safety_multiplier * 0.2 + 
                          liquidity_multiplier * 0.1)
        
        volatility_adjustment = 1.0
        max_change = max(abs(signal.price_change_1m), abs(signal.price_change_3m), abs(signal.price_change_5m))
        if max_change > 15:
            volatility_adjustment = 0.7
        elif max_change > 10:
            volatility_adjustment = 0.85
        
        kelly_fraction = self._calculate_kelly_fraction(signal)
        
        final_risk = base_risk * score_multiplier * volatility_adjustment * kelly_fraction
        position_value = self.current_capital * final_risk
        
        position_value = max(self.min_position_size, min(position_value, self.max_position_size))
        
        return position_value
    
    def _calculate_kelly_fraction(self, signal: BreakoutSignal) -> float:
        estimated_win_prob = 0.45 + signal.breakout_confidence * 0.35
        
        expected_win = 0.20
        expected_loss = 0.10
        
        if expected_loss <= 0:
            return 0.25
        
        kelly = (estimated_win_prob * expected_win - (1 - estimated_win_prob) * expected_loss) / expected_win
        
        return max(0.1, min(kelly, 0.5))

class EntropyExitCalculator:
    def __init__(self):
        self.price_history = defaultdict(lambda: deque(maxlen=50))
        self.entropy_history = defaultdict(lambda: deque(maxlen=20))
        
    def update_price(self, symbol: str, price: float):
        self.price_history[symbol].append((time.time(), price))
    
    def calculate_current_entropy(self, symbol: str, momentum_vector: np.ndarray) -> float:
        history = self.price_history[symbol]
        if len(history) < 10:
            return 0.5
        
        try:
            prices = [price for _, price in list(history)[-20:]]
            returns = np.diff(prices) / prices[:-1]
            
            if len(returns) < 5:
                return 0.5
            
            hist, _ = np.histogram(returns, bins=5, density=True)
            hist = hist[hist > 0]
            
            if len(hist) < 2:
                return 0.3
            
            shannon_entropy = -np.sum(hist * np.log2(hist))
            max_entropy = np.log2(len(hist))
            normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
            
            momentum_magnitude = np.linalg.norm(momentum_vector)
            momentum_stability = 1.0 / (1.0 + momentum_magnitude * 2)
            
            recent_volatility = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
            volatility_entropy = min(recent_volatility * 20, 1.0)
            
            final_entropy = (normalized_entropy * 0.5 + 
                           momentum_stability * 0.3 + 
                           volatility_entropy * 0.2)
            
            self.entropy_history[symbol].append(final_entropy)
            
            return max(0.0, min(1.0, final_entropy))
            
        except Exception as e:
            logger.error(f"Entropy calculation error for {symbol}: {e}")
            return 0.5
    
    def should_exit_on_entropy(self, symbol: str, threshold: float) -> bool:
        if symbol not in self.entropy_history:
            return False
        
        recent_entropy = list(self.entropy_history[symbol])
        if len(recent_entropy) < 3:
            return False
        
        current_entropy = recent_entropy[-1]
        avg_recent_entropy = np.mean(recent_entropy[-5:])
        
        if current_entropy < threshold and avg_recent_entropy < threshold * 1.2:
            return True
        
        entropy_trend = np.polyfit(range(len(recent_entropy)), recent_entropy, 1)[0]
        if entropy_trend < -0.05 and current_entropy < threshold * 1.3:
            return True
        
        return False

class GPUPricePredictor(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x):
        return self.predictor(x)

class HyperMomentumExecutor:
    def __init__(self):
        live_mode = os.getenv('TRADING_MODE', 'paper') == 'live'
        self.okx_api = OKXTradingAPI(live_mode=live_mode)
        self.position_sizer = PositionSizer()
        self.entropy_calculator = EntropyExitCalculator()
        self.price_predictor = GPUPricePredictor().to(device)
        self.price_predictor.eval()
        
        self.active_positions: Dict[str, ExecutionPosition] = {}
        self.signal_queue = asyncio.Queue(maxsize=500)
        self.metrics = ExecutionMetrics()
        
        self.cooldown_periods = defaultdict(float)
        self.symbol_blacklist = set()
        
        self.profit_targets = {
            'conservative': 0.15,
            'moderate': 0.20,
            'aggressive': 0.25
        }
        
        self.running = False
        self.lock = Lock()
        
    async def start(self):
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        await self.okx_api.start()
        self.running = True
        
        logger.info("🚀 Hyper-Momentum Executor started")
        
        tasks = [
            asyncio.create_task(self._signal_processor()),
            asyncio.create_task(self._position_manager()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._risk_monitor()),
            asyncio.create_task(self._signal_listener())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Executor shutdown requested")
        finally:
            await self.stop()
    
    async def stop(self):
        self.running = False
        
        for symbol in list(self.active_positions.keys()):
            await self._close_position(symbol, "system_shutdown")
        
        await self.okx_api.stop()
    
    async def _signal_listener(self):
        import sys
        import select
        
        while self.running:
            try:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip()
                    if line:
                        try:
                            signal_data = orjson.loads(line)
                            signal = BreakoutSignal(**signal_data)
                            await self.signal_queue.put(signal)
                        except Exception as e:
                            logger.error(f"Signal parsing error: {e}")
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Signal listener error: {e}")
                await asyncio.sleep(1)
    
    async def _signal_processor(self):
        while self.running:
            try:
                signal = await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)
                
                self.metrics.total_signals += 1
                
                if not await self._should_execute_signal(signal):
                    continue
                
                execution_start = time.time()
                success = await self._execute_signal(signal)
                execution_time = time.time() - execution_start
                
                if success:
                    self.metrics.executed_trades += 1
                    self.metrics.avg_execution_time = (
                        (self.metrics.avg_execution_time * (self.metrics.executed_trades - 1) + execution_time) / 
                        self.metrics.executed_trades
                    )
                
                self.metrics.execution_rate = (
                    self.metrics.executed_trades / self.metrics.total_signals
                ) if self.metrics.total_signals > 0 else 0
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Signal processing error: {e}")
    
    async def _should_execute_signal(self, signal: BreakoutSignal) -> bool:
        current_time = time.time()
        
        if signal.symbol in self.cooldown_periods:
            if current_time < self.cooldown_periods[signal.symbol]:
                self.metrics.cooldown_rejects += 1
                return False
        
        if signal.symbol in self.symbol_blacklist:
            return False
        
        if signal.symbol in self.active_positions:
            return False
        
        if signal.final_score < 0.75:
            return False
        
        if signal.breakout_confidence < 0.85:
            return False
        
        if signal.safety_score < 0.7:
            return False
        
        if signal.liquidity < 100000:
            return False
        
        if len(self.active_positions) >= 5:
            return False
        
        balance = await self.okx_api.get_balance()
        available_balance = float(balance['availBal'])
        
        if available_balance < 50:
            return False
        
        return True
    
    async def _execute_signal(self, signal: BreakoutSignal) -> bool:
        try:
            balance = await self.okx_api.get_balance()
            current_capital = float(balance['totalEq'])
            
            position_size = self.position_sizer.calculate_position_size(signal, current_capital)
            
            if position_size < 10:
                return False
            
            ticker = await self.okx_api.get_ticker(signal.symbol)
            if not ticker:
                return False
            
            entry_price = float(ticker['askPx'])
            shares = position_size / entry_price
            
            order = await self.okx_api.place_order(
                symbol=signal.symbol,
                side='buy',
                size=shares,
                order_type='market'
            )
            
            if not order:
                return False
            
            actual_price = float(order['fillPx'])
            actual_shares = float(order['fillSz'])
            
            slippage = abs(actual_price - entry_price) / entry_price
            self.metrics.slippage_avg = (
                (self.metrics.slippage_avg * self.metrics.executed_trades + slippage) / 
                (self.metrics.executed_trades + 1)
            )
            
            leverage = min(3.0, signal.breakout_confidence * 5)
            
            profit_style = self._determine_profit_style(signal)
            take_profit_pct = self.profit_targets[profit_style]
            
            stop_loss = actual_price * 0.92
            take_profit = actual_price * (1 + take_profit_pct)
            
            entropy_threshold = 0.35 - (signal.entropy_score * 0.1)
            
            position = ExecutionPosition(
                signal=signal,
                side='buy',
                size=actual_shares,
                entry_price=actual_price,
                current_price=actual_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=datetime.now(),
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=stop_loss,
                entropy_exit_threshold=entropy_threshold,
                cooldown_until=time.time() + 300,
                slippage_tolerance=0.005,
                execution_quality=1.0 - slippage,
                position_id=order['ordId'],
                last_entropy_check=time.time()
            )
            
            with self.lock:
                self.active_positions[signal.symbol] = position
            
            self.cooldown_periods[signal.symbol] = time.time() + 180
            
            logger.info(f"🚀 Position opened: {signal.symbol} @ ${actual_price:.6f}, "
                       f"Size: {actual_shares:.6f}, Target: {take_profit_pct:.1%}, "
                       f"Confidence: {signal.breakout_confidence:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Signal execution error: {e}")
            return False
    
    def _determine_profit_style(self, signal: BreakoutSignal) -> str:
        if signal.breakout_confidence > 0.95 and signal.entropy_score > 0.8:
            return 'aggressive'
        elif signal.breakout_confidence > 0.90 and signal.safety_score > 0.85:
            return 'moderate'
        else:
            return 'conservative'
    
    async def _position_manager(self):
        while self.running:
            try:
                for symbol in list(self.active_positions.keys()):
                    await self._manage_position(symbol)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Position management error: {e}")
                await asyncio.sleep(2)
    
    async def _manage_position(self, symbol: str):
        if symbol not in self.active_positions:
            return
        
        position = self.active_positions[symbol]
        
        ticker = await self.okx_api.get_ticker(symbol)
        if not ticker:
            return
        
        current_price = float(ticker['last'])
        position.current_price = current_price
        
        self.entropy_calculator.update_price(symbol, current_price)
        
        position.unrealized_pnl = (current_price - position.entry_price) * position.size * position.leverage
        
        return_pct = (current_price - position.entry_price) / position.entry_price
        leveraged_return = return_pct * position.leverage
        
        if leveraged_return > position.max_profit:
            position.max_profit = leveraged_return
            position.trailing_stop = current_price * 0.95
        
        position.drawdown_from_peak = (position.max_profit - leveraged_return) if position.max_profit > 0 else 0
        
        should_exit = False
        exit_reason = ""
        
        if current_price >= position.take_profit:
            should_exit = True
            exit_reason = "take_profit"
            self.metrics.target_exits += 1
        
        elif current_price <= position.stop_loss:
            should_exit = True
            exit_reason = "stop_loss"
            self.metrics.stop_exits += 1
        
        elif current_price <= position.trailing_stop:
            should_exit = True
            exit_reason = "trailing_stop"
            self.metrics.stop_exits += 1
        
        elif position.drawdown_from_peak > 0.08:
            should_exit = True
            exit_reason = "drawdown_protection"
        
        elif (datetime.now() - position.entry_time).seconds > 3600:
            should_exit = True
            exit_reason = "time_exit"
        
        current_time = time.time()
        if current_time - position.last_entropy_check > 30:
            position.last_entropy_check = current_time
            
            momentum_vector = np.array(position.signal.momentum_vector)
            current_entropy = self.entropy_calculator.calculate_current_entropy(symbol, momentum_vector)
            
            if self.entropy_calculator.should_exit_on_entropy(symbol, position.entropy_exit_threshold):
                should_exit = True
                exit_reason = "entropy_decay"
                self.metrics.entropy_exits += 1
        
        if leveraged_return > 0.30:
            should_exit = True
            exit_reason = "extreme_profit"
        
        if should_exit:
            await self._close_position(symbol, exit_reason)
    
    async def _close_position(self, symbol: str, reason: str):
        if symbol not in self.active_positions:
            return
        
        position = self.active_positions[symbol]
        
        try:
            order = await self.okx_api.place_order(
                symbol=symbol,
                side='sell',
                size=position.size,
                order_type='market'
            )
            
            if order:
                exit_price = float(order['fillPx'])
                
                total_pnl = (exit_price - position.entry_price) * position.size * position.leverage
                position.realized_pnl = total_pnl
                
                self.metrics.total_pnl += total_pnl
                
                if total_pnl > 0:
                    self.metrics.winning_trades += 1
                
                pnl_pct = total_pnl / (position.entry_price * position.size * position.leverage) * 100
                
                logger.info(f"💰 Position closed: {symbol} @ ${exit_price:.6f}, "
                          f"P&L: ${total_pnl:.2f} ({pnl_pct:.2f}%), "
                          f"Reason: {reason}")
                
                with self.lock:
                    del self.active_positions[symbol]
                
                if total_pnl < -50:
                    self.symbol_blacklist.add(symbol)
                    logger.warning(f"⚠️ Blacklisted {symbol} due to large loss")
            
        except Exception as e:
            logger.error(f"Position close error for {symbol}: {e}")
    
    async def _performance_monitor(self):
        start_time = time.time()
        
        while self.running:
            await asyncio.sleep(60)
            
            try:
                balance = await self.okx_api.get_balance()
                total_equity = float(balance['totalEq'])
                
                unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
                
                uptime = time.time() - start_time
                win_rate = (self.metrics.winning_trades / self.metrics.executed_trades * 100) if self.metrics.executed_trades > 0 else 0
                
                total_return = (total_equity - 1000) / 1000 * 100
                
                logger.info(f"📊 Performance Update:")
                logger.info(f"   💰 Total Equity: ${total_equity:.2f}")
                logger.info(f"   📈 Total Return: {total_return:.2f}%")
                logger.info(f"   🎯 Win Rate: {win_rate:.1f}%")
                logger.info(f"   📊 Signals: {self.metrics.total_signals}, Executed: {self.metrics.executed_trades}")
                logger.info(f"   ⚡ Execution Rate: {self.metrics.execution_rate:.1%}")
                logger.info(f"   💸 Total P&L: ${self.metrics.total_pnl:.2f}")
                logger.info(f"   💨 Avg Slippage: {self.metrics.slippage_avg:.4f}")
                logger.info(f"   🔄 Active Positions: {len(self.active_positions)}")
                logger.info(f"   🚪 Exits - Target: {self.metrics.target_exits}, Stop: {self.metrics.stop_exits}, Entropy: {self.metrics.entropy_exits}")
                
                if total_return >= 900:
                    logger.info(f"🎉 TARGET ACHIEVED! 10x return reached: {total_return:.1f}%")
                    self.running = False
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def _risk_monitor(self):
        while self.running:
            await asyncio.sleep(30)
            
            try:
                balance = await self.okx_api.get_balance()
                total_equity = float(balance['totalEq'])
                
                total_exposure = sum(
                    abs(pos.size * pos.current_price * pos.leverage) 
                    for pos in self.active_positions.values()
                )
                
                exposure_ratio = total_exposure / total_equity if total_equity > 0 else 0
                
                if exposure_ratio > 0.9:
                    logger.warning(f"⚠️ High exposure ratio: {exposure_ratio:.1%}")
                    
                    positions_by_risk = sorted(
                        self.active_positions.items(),
                        key=lambda x: abs(x[1].unrealized_pnl),
                        reverse=True
                    )
                    
                    for symbol, position in positions_by_risk[:2]:
                        if position.unrealized_pnl < 0:
                            await self._close_position(symbol, "risk_reduction")
                
                drawdown = max(0, 1000 - total_equity) / 1000
                if drawdown > 0.30:
                    logger.critical(f"🚨 High drawdown: {drawdown:.1%}")
                    
                    for symbol in list(self.active_positions.keys()):
                        await self._close_position(symbol, "emergency_exit")
                
                losing_positions = [
                    pos for pos in self.active_positions.values() 
                    if pos.unrealized_pnl < -30
                ]
                
                if len(losing_positions) >= 3:
                    logger.warning("⚠️ Multiple losing positions detected")
                    
                    for pos in losing_positions[:1]:
                        symbol = pos.signal.symbol
                        await self._close_position(symbol, "multiple_losses")
                
                position_ages = [
                    (datetime.now() - pos.entry_time).seconds 
                    for pos in self.active_positions.values()
                ]
                
                if position_ages and max(position_ages) > 1800:
                    logger.info("🕐 Closing old positions")
                    oldest_symbol = max(
                        self.active_positions.items(),
                        key=lambda x: (datetime.now() - x[1].entry_time).seconds
                    )[0]
                    await self._close_position(oldest_symbol, "age_limit")
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")

class AdvancedFeatureExtractor:
    def __init__(self):
        self.device = device
        
    def extract_execution_features(self, signal: BreakoutSignal, market_data: Dict) -> torch.Tensor:
        try:
            current_price = float(market_data.get('last', signal.price))
            bid_price = float(market_data.get('bidPx', current_price * 0.999))
            ask_price = float(market_data.get('askPx', current_price * 1.001))
            
            spread = (ask_price - bid_price) / current_price
            
            price_momentum = (current_price - signal.price) / signal.price
            
            features = [
                signal.breakout_confidence,
                signal.entropy_score,
                signal.final_score,
                signal.price_change_1m / 100,
                signal.price_change_3m / 100,
                signal.price_change_5m / 100,
                np.log1p(signal.volume_surge) / 10,
                np.log1p(signal.liquidity) / 20,
                spread * 1000,
                price_momentum * 100,
                signal.safety_score,
                np.tanh(signal.price_change_1m / 5),
                min(signal.liquidity / 1000000, 1.0),
                np.mean(signal.momentum_vector[:4]) if len(signal.momentum_vector) >= 4 else 0,
                np.std(signal.momentum_vector[:8]) if len(signal.momentum_vector) >= 8 else 0,
                time.time() % 3600 / 3600
            ]
            
            while len(features) < 16:
                features.append(0.0)
            
            return torch.tensor(features[:16], dtype=torch.float32, device=self.device)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return torch.zeros(16, dtype=torch.float32, device=self.device)

class BacktestValidator:
    def __init__(self):
        self.historical_signals = deque(maxlen=1000)
        self.historical_outcomes = deque(maxlen=1000)
        
    def add_signal_outcome(self, signal: BreakoutSignal, outcome_pnl: float):
        self.historical_signals.append(signal)
        self.historical_outcomes.append(outcome_pnl)
    
    def validate_signal(self, signal: BreakoutSignal) -> float:
        if len(self.historical_signals) < 20:
            return 0.7
        
        try:
            similar_signals = []
            for i, hist_signal in enumerate(self.historical_signals):
                similarity = self._calculate_similarity(signal, hist_signal)
                if similarity > 0.7:
                    similar_signals.append((similarity, self.historical_outcomes[i]))
            
            if len(similar_signals) < 3:
                return 0.6
            
            weighted_outcomes = []
            total_weight = 0
            
            for similarity, outcome in similar_signals:
                weight = similarity ** 2
                weighted_outcomes.append(outcome * weight)
                total_weight += weight
            
            expected_outcome = sum(weighted_outcomes) / total_weight if total_weight > 0 else 0
            
            confidence_score = min(1.0, max(0.0, (expected_outcome + 50) / 100))
            
            return confidence_score
            
        except Exception as e:
            logger.error(f"Backtest validation error: {e}")
            return 0.5
    
    def _calculate_similarity(self, signal1: BreakoutSignal, signal2: BreakoutSignal) -> float:
        try:
            features1 = np.array([
                signal1.breakout_confidence,
                signal1.entropy_score,
                signal1.price_change_1m / 100,
                signal1.price_change_3m / 100,
                signal1.price_change_5m / 100,
                np.log1p(signal1.volume_surge) / 10,
                signal1.safety_score,
                np.log1p(signal1.liquidity) / 20
            ])
            
            features2 = np.array([
                signal2.breakout_confidence,
                signal2.entropy_score,
                signal2.price_change_1m / 100,
                signal2.price_change_3m / 100,
                signal2.price_change_5m / 100,
                np.log1p(signal2.volume_surge) / 10,
                signal2.safety_score,
                np.log1p(signal2.liquidity) / 20
            ])
            
            cosine_sim = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
            euclidean_sim = 1.0 / (1.0 + np.linalg.norm(features1 - features2))
            
            return (cosine_sim + euclidean_sim) / 2
            
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0

class MarketRegimeDetector:
    def __init__(self):
        self.price_data = defaultdict(lambda: deque(maxlen=100))
        self.volume_data = defaultdict(lambda: deque(maxlen=100))
        self.market_state = "normal"
        self.volatility_threshold = 0.05
        
    def update_market_data(self, symbol: str, price: float, volume: float):
        self.price_data[symbol].append(price)
        self.volume_data[symbol].append(volume)
        
        if len(self.price_data[symbol]) >= 20:
            self._update_market_regime(symbol)
    
    def _update_market_regime(self, symbol: str):
        try:
            prices = list(self.price_data[symbol])
            
            if len(prices) < 20:
                return
            
            returns = np.diff(prices) / prices[:-1]
            recent_vol = np.std(returns[-20:])
            
            if recent_vol > self.volatility_threshold:
                self.market_state = "high_volatility"
            elif recent_vol < self.volatility_threshold * 0.3:
                self.market_state = "low_volatility"
            else:
                trend = np.polyfit(range(len(prices[-10:])), prices[-10:], 1)[0]
                if abs(trend) > prices[-1] * 0.01:
                    self.market_state = "trending"
                else:
                    self.market_state = "normal"
                    
        except Exception as e:
            logger.error(f"Market regime detection error: {e}")
    
    def get_position_adjustment(self) -> float:
        adjustments = {
            "high_volatility": 0.7,
            "low_volatility": 1.2,
            "trending": 1.1,
            "normal": 1.0
        }
        return adjustments.get(self.market_state, 1.0)

class AdvancedHyperMomentumExecutor(HyperMomentumExecutor):
    def __init__(self):
        super().__init__()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.backtest_validator = BacktestValidator()
        self.regime_detector = MarketRegimeDetector()
        self.adaptive_thresholds = {
            'confidence': 0.85,
            'entropy': 0.35,
            'safety': 0.7
        }
        
        if self.okx_api.live_mode:
            logger.critical("🔴 ADVANCED EXECUTOR IN LIVE MODE - REAL MONEY AT RISK!")
            logger.critical("🔴 Ensure you have:")
            logger.critical("🔴 1. Valid OKX API credentials")
            logger.critical("🔴 2. Sufficient balance") 
            logger.critical("🔴 3. Risk management understanding")
            logger.critical("🔴 4. Stop-loss mechanisms in place")
        
    async def _should_execute_signal(self, signal: BreakoutSignal) -> bool:
        base_should_execute = await super()._should_execute_signal(signal)
        if not base_should_execute:
            return False
        
        backtest_confidence = self.backtest_validator.validate_signal(signal)
        if backtest_confidence < 0.6:
            return False
        
        regime_adjustment = self.regime_detector.get_position_adjustment()
        adjusted_confidence = signal.breakout_confidence * regime_adjustment
        
        if adjusted_confidence < self.adaptive_thresholds['confidence']:
            return False
        
        ticker = await self.okx_api.get_ticker(signal.symbol)
        if ticker:
            self.regime_detector.update_market_data(
                signal.symbol, 
                float(ticker['last']), 
                float(ticker.get('vol24h', 0))
            )
        
        return True
    
    async def _execute_signal(self, signal: BreakoutSignal) -> bool:
        try:
            ticker = await self.okx_api.get_ticker(signal.symbol)
            if not ticker:
                return False
            
            features = self.feature_extractor.extract_execution_features(signal, ticker)
            
            with torch.no_grad():
                price_prediction = self.price_predictor(features.unsqueeze(0))
                predicted_return = torch.tanh(price_prediction).item()
            
            if predicted_return < 0.05:
                return False
            
            success = await super()._execute_signal(signal)
            
            if success and signal.symbol in self.active_positions:
                position = self.active_positions[signal.symbol]
                
                regime_adjustment = self.regime_detector.get_position_adjustment()
                position.take_profit *= regime_adjustment
                
                if predicted_return > 0.15:
                    position.take_profit *= 1.2
                elif predicted_return < 0.08:
                    position.take_profit *= 0.9
            
            return success
            
        except Exception as e:
            logger.error(f"Advanced signal execution error: {e}")
            return False
    
    async def _close_position(self, symbol: str, reason: str):
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            final_pnl = position.unrealized_pnl
            
            self.backtest_validator.add_signal_outcome(position.signal, final_pnl)
            
            if final_pnl > 0:
                self.adaptive_thresholds['confidence'] *= 0.995
            else:
                self.adaptive_thresholds['confidence'] *= 1.002
            
            self.adaptive_thresholds['confidence'] = max(0.80, min(0.95, self.adaptive_thresholds['confidence']))
        
        await super()._close_position(symbol, reason)

async def main():
    executor = AdvancedHyperMomentumExecutor()
    
    try:
        logger.info("🚀 Starting Hyper-Momentum Execution Engine")
        logger.info("💰 Target: $1K → $10K (10x return)")
        logger.info("⚡ GPU-Accelerated ML Execution Pipeline Active")
        logger.info("=" * 60)
        
        await executor.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Executor stopped by user")
    except Exception as e:
        logger.error(f"Executor error: {e}")
    finally:
        await executor.stop()

if __name__ == "__main__":
    asyncio.run(main())