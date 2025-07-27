#!/usr/bin/env python3

import asyncio
import aiohttp
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import json
import uvloop

@dataclass
class OKXMarketData:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    change_24h: float
    high_24h: float
    low_24h: float
    timestamp: float

@dataclass
class FastFeatures:
    price_change: float
    volatility: float
    momentum: float
    rsi: float
    spread: float
    volume_surge: float
    target: int = 0

@dataclass
class TradingParams:
    confidence: float = 0.75
    position_size: float = 0.08
    stop_loss: float = 0.018
    take_profit: float = 0.035
    generation: int = 0

class OKXRealTimeAPI:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.session = None
        self.base_url = "https://www.okx.com"
        self.request_count = 0
        self.last_reset = time.time()
        
    async def start(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin'
        }
        
        connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            use_dns_cache=True,
            family=0
        )
        
        timeout = aiohttp.ClientTimeout(total=3, connect=1)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
        print("🚀 OKX API connection established")
        
    async def stop(self):
        if self.session:
            await self.session.close()
    
    async def get_okx_tickers(self) -> Dict[str, OKXMarketData]:
        try:
            start_time = time.perf_counter_ns()
            
            # OKX uses different symbol format
            okx_symbols = [s.replace('USDT', '-USDT') for s in self.symbols]
            symbols_param = ','.join(okx_symbols)
            
            url = f"{self.base_url}/api/v5/market/tickers?instType=SPOT"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    print(f"❌ OKX API error: {response.status}")
                    return {}
                
                response_data = await response.json()
                
                if 'data' not in response_data:
                    print("❌ OKX: No data field in response")
                    return {}
                
                result = {}
                data = response_data['data']
                
                for ticker in data:
                    inst_id = ticker['instId']
                    # Convert back to our format
                    our_symbol = inst_id.replace('-USDT', 'USDT')
                    
                    if our_symbol in self.symbols:
                        price = float(ticker['last'])
                        bid = float(ticker['bidPx']) if ticker['bidPx'] else price * 0.9995
                        ask = float(ticker['askPx']) if ticker['askPx'] else price * 1.0005
                        
                        result[our_symbol] = OKXMarketData(
                            symbol=our_symbol,
                            price=price,
                            bid=bid,
                            ask=ask,
                            volume=float(ticker['vol24h']) if ticker['vol24h'] else 0,
                            change_24h=float(ticker['chgUtc0']) * 100 if ticker['chgUtc0'] else 0,
                            high_24h=float(ticker['high24h']) if ticker['high24h'] else price,
                            low_24h=float(ticker['low24h']) if ticker['low24h'] else price,
                            timestamp=time.perf_counter_ns()
                        )
                
                latency_ns = time.perf_counter_ns() - start_time
                latency_us = latency_ns / 1000
                
                if result:
                    price_updates = []
                    for symbol, data in list(result.items())[:3]:
                        trend = "📈" if data.change_24h > 0 else "📉" if data.change_24h < 0 else "➡️"
                        price_updates.append(f"{symbol}:{trend}${data.price:.4f}({data.change_24h:+.2f}%)")
                    
                    print(f"🔥 OKX LIVE: {' | '.join(price_updates)} | {latency_us:.0f}μs")
                
                return result
                
        except asyncio.TimeoutError:
            print("⚠️ OKX timeout")
            return {}
        except Exception as e:
            print(f"❌ OKX error: {e}")
            return {}
    
    async def get_okx_order_book(self, symbol: str) -> Optional[tuple]:
        try:
            okx_symbol = symbol.replace('USDT', '-USDT')
            url = f"{self.base_url}/api/v5/market/books?instId={okx_symbol}&sz=5"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and data['data']:
                        book = data['data'][0]
                        if 'bids' in book and 'asks' in book and book['bids'] and book['asks']:
                            best_bid = float(book['bids'][0][0])
                            best_ask = float(book['asks'][0][0])
                            return best_bid, best_ask
        except:
            pass
        return None

class UltraFastML:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.ready = False
        self.prediction_cache = {}
        
    def initialize(self):
        try:
            import tensorflow as tf
            
            # GPU optimization
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("🚀 GPU acceleration active")
            
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(5,)),
                tf.keras.layers.Dense(12, activation='relu'),
                tf.keras.layers.Dense(6, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            
            return True
            
        except Exception as e:
            print(f"❌ ML init failed: {e}")
            return False
    
    def extract_features(self, prices: np.ndarray, volume_data: List[float] = None) -> Optional[FastFeatures]:
        if len(prices) < 15:
            return None
        
        try:
            returns = np.diff(prices) / prices[:-1]
            returns = np.nan_to_num(returns, 0.0)
            
            price_change = returns[-1]
            volatility = np.std(returns[-8:])
            momentum = np.mean(returns[-3:])
            
            # RSI calculation
            gains = np.where(returns[-10:] > 0, returns[-10:], 0)
            losses = np.where(returns[-10:] < 0, -returns[-10:], 0)
            avg_gain = np.mean(gains) if np.sum(gains) > 0 else 0.001
            avg_loss = np.mean(losses) if np.sum(losses) > 0 else 0.001
            rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            
            # Spread (we'll calculate from bid/ask later)
            spread = 0.001
            
            # Volume surge detection
            volume_surge = 1.0
            if volume_data and len(volume_data) >= 5:
                recent_vol = np.mean(volume_data[-2:])
                avg_vol = np.mean(volume_data[-5:])
                if avg_vol > 0:
                    volume_surge = recent_vol / avg_vol
            
            return FastFeatures(
                price_change=np.clip(price_change, -0.1, 0.1),
                volatility=np.clip(volatility, 0.0005, 0.05),
                momentum=np.clip(momentum, -0.03, 0.03),
                rsi=np.clip(rsi / 100.0, 0.0, 1.0),
                spread=np.clip(spread, 0.0001, 0.01),
                volume_surge=np.clip(volume_surge, 0.1, 5.0)
            )
            
        except:
            return None
    
    def train_fast(self, features_list: List[FastFeatures]) -> bool:
        if len(features_list) < 25:
            return False
        
        try:
            import tensorflow as tf
            
            X = np.array([[
                f.price_change, f.volatility, f.momentum, f.rsi, f.spread
            ] for f in features_list])
            
            y = np.array([f.target for f in features_list])
            
            # Balance classes
            unique_targets = np.unique(y)
            if len(unique_targets) < 2:
                return False
            
            balanced_X = []
            balanced_y = []
            min_count = min([np.sum(y == target) for target in unique_targets])
            min_count = max(min_count, 8)
            
            for target in unique_targets:
                target_mask = y == target
                target_X = X[target_mask]
                target_y = y[target_mask]
                
                if len(target_X) >= min_count:
                    indices = np.random.choice(len(target_X), min_count, replace=False)
                else:
                    indices = np.random.choice(len(target_X), min_count, replace=True)
                
                balanced_X.extend(target_X[indices])
                balanced_y.extend(target_y[indices])
            
            X_balanced = np.array(balanced_X)
            y_balanced = np.array(balanced_y)
            
            if not hasattr(self.scaler, 'scale_'):
                X_scaled = self.scaler.fit_transform(X_balanced)
            else:
                X_scaled = self.scaler.transform(X_balanced)
            
            history = self.model.fit(
                X_scaled, y_balanced,
                epochs=5,
                batch_size=16,
                validation_split=0.2,
                verbose=0
            )
            
            self.ready = True
            
            val_acc = history.history.get('val_accuracy', [0])[-1]
            train_acc = history.history.get('accuracy', [0])[-1]
            
            status = "🚀" if val_acc > 0.6 else "📈" if val_acc > 0.5 else "🔄"
            overfitting = " ⚠️" if (train_acc - val_acc) > 0.2 else ""
            
            print(f"🧠 {status} OKX ML: T:{train_acc:.3f} V:{val_acc:.3f}{overfitting} | {len(balanced_X)} samples")
            
            return True
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def predict_fast(self, features: FastFeatures) -> tuple[int, float]:
        if not self.ready:
            return 1, 0.5
        
        try:
            cache_key = hash((round(features.price_change, 6), round(features.volatility, 6)))
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            X = np.array([[
                features.price_change, features.volatility, features.momentum,
                features.rsi, features.spread
            ]])
            
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled, verbose=0)
            
            predicted_class = int(np.argmax(pred[0]))
            confidence = float(np.max(pred[0]))
            
            result = (predicted_class, confidence)
            self.prediction_cache[cache_key] = result
            
            if len(self.prediction_cache) > 500:
                self.prediction_cache.clear()
            
            return result
            
        except:
            return 1, 0.5

class OKXTradingSystem:
    def __init__(self, symbols: List[str], capital: float = 1000):
        self.symbols = symbols
        self.initial_capital = capital
        self.current_capital = capital
        
        self.okx_api = OKXRealTimeAPI(symbols)
        self.ml_engine = UltraFastML()
        
        self.price_history = {s: deque(maxlen=30) for s in symbols}
        self.volume_history = {s: deque(maxlen=10) for s in symbols}
        self.feature_buffer = deque(maxlen=500)
        
        self.positions = {}
        self.trades = []
        self.running = False
        
        self.params = TradingParams()
        self.last_trade_times = {s: 0 for s in symbols}
        
    async def start(self):
        try:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            print("🚀 uvloop acceleration enabled")
        except:
            print("⚠️ uvloop not available")
        
        print("🔥 OKX REAL-TIME TRADING SYSTEM")
        print("⚡ Pure OKX • Real trades • GPU ML")
        print("=" * 50)
        
        if not self.ml_engine.initialize():
            print("❌ ML failed to start")
            return
        
        await self.okx_api.start()
        self.running = True
        
        tasks = [
            asyncio.create_task(self.okx_data_feed()),
            asyncio.create_task(self.ml_training_loop()),
            asyncio.create_task(self.okx_trading_engine()),
            asyncio.create_task(self.performance_monitor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n🛑 OKX trading stopped")
        finally:
            await self.cleanup()
    
    async def okx_data_feed(self):
        print("📡 OKX real-time feed starting...")
        
        while self.running:
            try:
                market_data = await self.okx_api.get_okx_tickers()
                
                if market_data:
                    for symbol, data in market_data.items():
                        old_price = self.price_history[symbol][-1] if self.price_history[symbol] else None
                        
                        self.price_history[symbol].append(data.price)
                        self.volume_history[symbol].append(data.volume)
                        
                        if len(self.price_history[symbol]) >= 15:
                            prices = np.array(list(self.price_history[symbol]))
                            volumes = list(self.volume_history[symbol])
                            features = self.ml_engine.extract_features(prices, volumes)
                            
                            if features:
                                # Real spread from bid/ask
                                if data.bid > 0 and data.ask > 0:
                                    features.spread = (data.ask - data.bid) / data.price
                                
                                # Label generation
                                if len(self.price_history[symbol]) >= 20:
                                    past_price = list(self.price_history[symbol])[-8]
                                    current_price = data.price
                                    future_return = (current_price - past_price) / past_price
                                    
                                    if future_return > 0.008:  # 0.8% gain
                                        features.target = 2  # Buy
                                    elif future_return < -0.008:  # 0.8% loss
                                        features.target = 0  # Sell
                                    else:
                                        features.target = 1  # Hold
                                
                                self.feature_buffer.append(features)
                
                await asyncio.sleep(1.5)  # OKX rate limit friendly
                
            except Exception as e:
                print(f"❌ Data feed error: {e}")
                await asyncio.sleep(5)
    
    async def ml_training_loop(self):
        while self.running:
            try:
                if len(self.feature_buffer) >= 40:
                    recent_features = list(self.feature_buffer)[-80:]
                    
                    # Check class distribution
                    targets = [f.target for f in recent_features]
                    unique_targets = set(targets)
                    
                    if len(unique_targets) >= 2:
                        self.ml_engine.train_fast(recent_features)
                
                await asyncio.sleep(15)
                
            except Exception as e:
                print(f"❌ Training error: {e}")
                await asyncio.sleep(30)
    
    async def okx_trading_engine(self):
        print("⚡ OKX trading engine active")
        
        while self.running:
            try:
                if not self.ml_engine.ready:
                    await asyncio.sleep(1)
                    continue
                
                current_time = time.time()
                
                # Check existing positions
                for symbol in list(self.positions.keys()):
                    await self.manage_okx_position(symbol)
                
                if len(self.positions) >= 2:
                    await asyncio.sleep(0.5)
                    continue
                
                # Look for new trades
                market_data = await self.okx_api.get_okx_tickers()
                
                for symbol in self.symbols:
                    if symbol in self.positions or symbol not in market_data:
                        continue
                    
                    if current_time - self.last_trade_times[symbol] < 120:  # 2 min cooldown
                        continue
                    
                    if len(self.price_history[symbol]) < 15:
                        continue
                    
                    prices = np.array(list(self.price_history[symbol]))
                    volumes = list(self.volume_history[symbol])
                    features = self.ml_engine.extract_features(prices, volumes)
                    
                    if not features:
                        continue
                    
                    # Update spread with real data
                    data = market_data[symbol]
                    if data.bid > 0 and data.ask > 0:
                        features.spread = (data.ask - data.bid) / data.price
                    
                    predicted_class, confidence = self.ml_engine.predict_fast(features)
                    
                    if predicted_class != 1 and confidence >= self.params.confidence:
                        side = "buy" if predicted_class == 2 else "sell"
                        await self.execute_okx_trade(symbol, side, confidence, data)
                        self.last_trade_times[symbol] = current_time
                
                await asyncio.sleep(0.8)
                
            except Exception as e:
                print(f"❌ Trading error: {e}")
                await asyncio.sleep(2)
    
    async def execute_okx_trade(self, symbol: str, side: str, confidence: float, market_data: OKXMarketData):
        # Get precise bid/ask
        book_data = await self.okx_api.get_okx_order_book(symbol)
        if book_data:
            bid, ask = book_data
        else:
            bid, ask = market_data.bid, market_data.ask
        
        entry_price = ask if side == "buy" else bid
        
        position_size = self.current_capital * self.params.position_size * confidence
        position_size = max(20, min(position_size, self.current_capital * 0.15))
        
        shares = position_size / entry_price
        
        # Dynamic stops based on volatility and spread
        spread_factor = (ask - bid) / market_data.price
        volatility_factor = abs(market_data.change_24h) / 100
        
        stop_pct = self.params.stop_loss * (1 + spread_factor * 8 + volatility_factor * 2)
        profit_pct = self.params.take_profit * (1 + spread_factor * 5 + volatility_factor * 1.5)
        
        if side == "buy":
            stop_price = entry_price * (1 - stop_pct)
            profit_price = entry_price * (1 + profit_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)
            profit_price = entry_price * (1 - profit_pct)
        
        position = {
            "symbol": symbol,
            "side": side,
            "shares": shares,
            "entry_price": entry_price,
            "entry_time": time.time(),
            "stop_loss": stop_price,
            "take_profit": profit_price,
            "confidence": confidence,
            "spread": spread_factor,
            "volatility": volatility_factor
        }
        
        self.positions[symbol] = position
        self.current_capital -= position_size
        
        print(f"\n🔥 OKX TRADE EXECUTED")
        print(f"   {side.upper()} {symbol} @ ${entry_price:.4f}")
        print(f"   💰 Size: ${position_size:.0f} | Conf: {confidence:.1%}")
        print(f"   📊 Spread: {spread_factor*100:.3f}% | Vol: {volatility_factor*100:.1f}%")
        print(f"   🎯 TP: ${profit_price:.4f} | SL: ${stop_price:.4f}")
    
    async def manage_okx_position(self, symbol: str):
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        market_data = await self.okx_api.get_okx_tickers()
        if symbol not in market_data:
            return
        
        data = market_data[symbol]
        current_price = data.bid if position["side"] == "buy" else data.ask
        
        if position["side"] == "buy":
            pnl = (current_price - position["entry_price"]) * position["shares"]
            pnl_pct = (current_price - position["entry_price"]) / position["entry_price"] * 100
        else:
            pnl = (position["entry_price"] - current_price) * position["shares"]
            pnl_pct = (position["entry_price"] - current_price) / position["entry_price"] * 100
        
        time_held = time.time() - position["entry_time"]
        
        should_exit = False
        reason = ""
        
        # Standard exits
        if position["side"] == "buy":
            if current_price >= position["take_profit"]:
                should_exit, reason = True, "take_profit"
            elif current_price <= position["stop_loss"]:
                should_exit, reason = True, "stop_loss"
        else:
            if current_price <= position["take_profit"]:
                should_exit, reason = True, "take_profit"
            elif current_price >= position["stop_loss"]:
                should_exit, reason = True, "stop_loss"
        
        # Time exit
        if not should_exit and time_held > 600:  # 10 minutes max
            should_exit, reason = True, "time_exit"
        
        # Trailing stop for profits
        if not should_exit and pnl > 0 and pnl_pct > 1.5:
            trail_pct = self.params.stop_loss * 0.6
            if position["side"] == "buy":
                new_stop = current_price * (1 - trail_pct)
                if new_stop > position["stop_loss"]:
                    position["stop_loss"] = new_stop
            else:
                new_stop = current_price * (1 + trail_pct)
                if new_stop < position["stop_loss"]:
                    position["stop_loss"] = new_stop
        
        if should_exit:
            await self.close_okx_position(symbol, reason, current_price)
    
    async def close_okx_position(self, symbol: str, reason: str, exit_price: float):
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        if position["side"] == "buy":
            pnl = (exit_price - position["entry_price"]) * position["shares"]
            pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
        else:
            pnl = (position["entry_price"] - exit_price) * position["shares"]
            pnl_pct = (position["entry_price"] - exit_price) / position["entry_price"] * 100
        
        position_value = position["shares"] * position["entry_price"]
        self.current_capital += position_value + pnl
        
        duration = time.time() - position["entry_time"]
        
        trade = {
            "symbol": symbol,
            "side": position["side"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "duration": duration,
            "reason": reason,
            "confidence": position["confidence"]
        }
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        result = "💚" if pnl > 0 else "💔"
        print(f"\n{result} OKX EXIT: {symbol}")
        print(f"   💰 P&L: ${pnl:.2f} ({pnl_pct:.1f}%) | {duration:.0f}s")
        print(f"   🔥 Reason: {reason} | Conf: {position['confidence']:.1%}")
    
    async def performance_monitor(self):
        while self.running:
            await asyncio.sleep(45)
            
            if len(self.trades) > 0:
                total_pnl = sum(t['pnl'] for t in self.trades)
                total_return = (self.current_capital + total_pnl - self.initial_capital) / self.initial_capital * 100
                
                recent = self.trades[-8:] if len(self.trades) >= 8 else self.trades
                win_rate = sum(1 for t in recent if t['pnl'] > 0) / len(recent) * 100
                
                print(f"\n🔥 OKX PERFORMANCE")
                print(f"   💰 Capital: ${self.current_capital:.2f}")
                print(f"   📈 Return: {total_return:.1f}%")
                print(f"   🎯 Win Rate: {win_rate:.1f}% ({len(recent)} trades)")
                print(f"   📊 Active: {len(self.positions)} positions")
                print(f"   🧠 Features: {len(self.feature_buffer)}")
    
    async def cleanup(self):
        self.running = False
        
        # Close all positions
        market_data = await self.okx_api.get_okx_tickers()
        for symbol in list(self.positions.keys()):
            if symbol in market_data:
                data = market_data[symbol]
                current_price = data.bid if self.positions[symbol]["side"] == "buy" else data.ask
                await self.close_okx_position(symbol, "shutdown", current_price)
        
        await self.okx_api.stop()
        
        if self.trades:
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
            win_rate = sum(1 for t in self.trades if t['pnl'] > 0) / len(self.trades) * 100
            
            print(f"\n🔥 OKX FINAL RESULTS")
            print("=" * 40)
            print(f"💰 Final: ${self.current_capital:.2f}")
            print(f"🚀 Return: {total_return:.1f}%")
            print(f"📊 Trades: {len(self.trades)}")
            print(f"💚 Win Rate: {win_rate:.1f}%")
            print(f"🔥 Exchange: OKX")

async def main():
    print("🔥 OKX PURE TRADING SYSTEM")
    print("⚡ Real OKX data • Paper trading • GPU ML")
    print("=" * 50)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    system = OKXTradingSystem(symbols, 1000.0)
    
    await system.start()

if __name__ == "__main__":
    asyncio.run(main())