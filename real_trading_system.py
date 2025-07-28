#!/usr/bin/env python3
"""
🚀 REAL CRYPTO TRADING SYSTEM - FIXED VERSION
Live data only - No simulations or fake data
Two modes: Live execution or Paper execution (real prices, terminal trades)
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LivePosition:
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    stop_loss: float
    take_profit: float

@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    entry_time: datetime
    exit_time: datetime
    reason: str

class RealMarketDataFeed:
    """Real-time market data from actual exchanges"""
    
    def __init__(self):
        self.session = None
        self.price_cache = {}
        self.callbacks = []
        
    async def start(self):
        connector = aiohttp.TCPConnector(limit=50, keepalive_timeout=60)
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
    async def stop(self):
        if self.session:
            await self.session.close()
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    async def get_live_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get real live prices from OKX exchange - FIXED VERSION"""
        try:
            # Convert symbols to OKX format
            okx_symbols = [s.replace('USDT', '-USDT') for s in symbols]
            
            url = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"OKX API error: {response.status}")
                    return {}
                
                data = await response.json()
                
                if 'data' not in data:
                    logger.warning("No data field in OKX response")
                    return {}
                
                result = {}
                for ticker in data['data']:
                    inst_id = ticker['instId']
                    symbol = inst_id.replace('-USDT', 'USDT')
                    
                    if symbol in symbols:
                        # FIXED: Handle missing or different field names safely
                        try:
                            # Try different possible field names for 24h change
                            change_24h = 0.0
                            for change_field in ['chgUtc0', 'change24h', 'priceChangePercent', 'chg24h']:
                                if change_field in ticker and ticker[change_field]:
                                    change_24h = float(ticker[change_field]) * 100
                                    break
                            
                            price_data = {
                                'symbol': symbol,
                                'price': float(ticker['last']) if ticker.get('last') else 0.0,
                                'bid': float(ticker['bidPx']) if ticker.get('bidPx') else 0.0,
                                'ask': float(ticker['askPx']) if ticker.get('askPx') else 0.0,
                                'volume': float(ticker['vol24h']) if ticker.get('vol24h') else 0.0,
                                'change_24h': change_24h,
                                'timestamp': time.time()
                            }
                            
                            # Only add if we have a valid price
                            if price_data['price'] > 0:
                                result[symbol] = price_data
                                
                                # Update cache
                                self.price_cache[symbol] = price_data
                                
                                # Notify callbacks
                                for callback in self.callbacks:
                                    await callback(price_data)
                        
                        except (ValueError, TypeError, KeyError) as e:
                            logger.warning(f"Error processing ticker for {symbol}: {e}")
                            continue
                
                logger.info(f"📡 Retrieved prices for {len(result)} symbols")
                return result
                
        except Exception as e:
            logger.error(f"Failed to get live prices: {e}")
            # Return cached data if available
            if self.price_cache:
                logger.info("Using cached price data")
                return self.price_cache
            return {}

class RealTechnicalAnalysis:
    """Technical analysis using only real market data"""
    
    def __init__(self):
        self.price_history = {symbol: deque(maxlen=200) for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']}
    
    def update_price(self, symbol: str, price: float):
        """Update price history with real data"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=200)
            
        self.price_history[symbol].append({
            'price': price,
            'timestamp': time.time()
        })
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """Calculate RSI using real price history"""
        if symbol not in self.price_history:
            return 50.0
        
        prices = [p['price'] for p in list(self.price_history[symbol])]
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_moving_averages(self, symbol: str, short: int = 20, long: int = 50) -> Tuple[float, float]:
        """Calculate moving averages using real data"""
        if symbol not in self.price_history:
            return 0.0, 0.0
        
        prices = [p['price'] for p in list(self.price_history[symbol])]
        
        if len(prices) < long:
            return 0.0, 0.0
        
        sma_short = np.mean(prices[-short:]) if len(prices) >= short else 0.0
        sma_long = np.mean(prices[-long:]) if len(prices) >= long else 0.0
        
        return sma_short, sma_long
    
    def generate_trading_signal(self, symbol: str) -> Tuple[str, float]:
        """Generate trading signal based on real technical analysis"""
        rsi = self.calculate_rsi(symbol)
        sma_20, sma_50 = self.calculate_moving_averages(symbol)
        
        if len(list(self.price_history[symbol])) < 50:
            return "hold", 0.0
        
        current_price = list(self.price_history[symbol])[-1]['price']
        
        # RSI + Moving Average Strategy
        signal = "hold"
        confidence = 0.0
        
        if rsi < 30 and sma_20 > sma_50 and current_price > sma_20:
            signal = "buy"
            confidence = min(0.8, (30 - rsi) / 30 + 0.3)
        elif rsi > 70 and sma_20 < sma_50 and current_price < sma_20:
            signal = "sell"
            confidence = min(0.8, (rsi - 70) / 30 + 0.3)
        elif sma_20 > sma_50 * 1.01 and rsi > 50:
            signal = "buy"
            confidence = 0.6
        elif sma_20 < sma_50 * 0.99 and rsi < 50:
            signal = "sell"
            confidence = 0.6
        
        return signal, confidence

class LiveTradingEngine:
    """Execute trades on real exchange or paper terminal"""
    
    def __init__(self, mode: str = "paper", initial_capital: float = 1000.0):
        self.mode = mode  # "live" or "paper"
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        self.positions: Dict[str, LivePosition] = {}
        self.trades: List[Trade] = []
        
        # Only for paper mode - track simulated balance
        self.paper_balance = initial_capital if mode == "paper" else 0
        
        logger.info(f"Trading engine initialized in {mode.upper()} mode")
        if mode == "paper":
            logger.info(f"Paper trading with ${initial_capital:,.2f} virtual capital")
    
    async def execute_trade(self, symbol: str, side: str, size: float, price: float) -> bool:
        """Execute trade in live or paper mode"""
        
        if self.mode == "live":
            return await self._execute_live_trade(symbol, side, size, price)
        else:
            return await self._execute_paper_trade(symbol, side, size, price)
    
    async def _execute_live_trade(self, symbol: str, side: str, size: float, price: float) -> bool:
        """Execute real trade on exchange"""
        logger.critical(f"🔴 LIVE TRADE: {side.upper()} {size:.6f} {symbol} @ ${price:.2f}")
        logger.critical("🔴 THIS WOULD PLACE A REAL ORDER WITH REAL MONEY!")
        
        # In a real implementation, this would connect to exchange API
        # For safety, we're not implementing the actual live trading here
        # You would need to add your exchange API integration
        
        return False  # Always return False for safety
    
    async def _execute_paper_trade(self, symbol: str, side: str, size: float, price: float) -> bool:
        """Execute paper trade (real prices, terminal output)"""
        
        trade_value = size * price
        
        if side == "buy":
            if self.paper_balance < trade_value:
                logger.warning(f"Insufficient balance for {symbol} buy: ${trade_value:.2f}")
                return False
            
            self.paper_balance -= trade_value
            
            position = LivePosition(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
                entry_time=datetime.now(),
                stop_loss=price * 0.95,  # 5% stop loss
                take_profit=price * 1.10  # 10% take profit
            )
            
            self.positions[symbol] = position
            
            print(f"📈 PAPER BUY: {size:.6f} {symbol} @ ${price:.2f}")
            print(f"💰 Balance: ${self.paper_balance:.2f}")
            
        elif side == "sell":
            if symbol not in self.positions:
                logger.warning(f"No position to sell for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            pnl = (price - position.entry_price) * position.size
            self.paper_balance += (position.size * price)
            
            trade = Trade(
                symbol=symbol,
                side="sell",
                entry_price=position.entry_price,
                exit_price=price,
                size=position.size,
                pnl=pnl,
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                reason="manual_sell"
            )
            
            self.trades.append(trade)
            del self.positions[symbol]
            
            print(f"📉 PAPER SELL: {position.size:.6f} {symbol} @ ${price:.2f}")
            print(f"💰 P&L: ${pnl:.2f} | Balance: ${self.paper_balance:.2f}")
        
        return True
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized = sum(trade.pnl for trade in self.trades)
        
        return {
            'mode': self.mode,
            'balance': self.paper_balance if self.mode == "paper" else "N/A",
            'positions': len(self.positions),
            'trades': len(self.trades),
            'unrealized_pnl': total_unrealized,
            'realized_pnl': total_realized,
            'total_return': ((self.paper_balance + total_unrealized - self.initial_capital) / self.initial_capital * 100) if self.mode == "paper" else 0
        }

class RealTradingSystem:
    """Main trading system using only real data"""
    
    def __init__(self, mode: str = "paper"):
        self.mode = mode
        self.data_feed = RealMarketDataFeed()
        self.technical_analysis = RealTechnicalAnalysis()
        self.trading_engine = LiveTradingEngine(mode)
        
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
        self.running = False
        
        # Performance tracking
        self.last_signal_time = {}
        self.signal_count = 0
        
    async def start(self):
        """Start the real trading system"""
        
        print(f"🚀 STARTING REAL TRADING SYSTEM")
        print(f"📊 Mode: {self.mode.upper()}")
        print(f"💰 Initial Capital: ${self.trading_engine.initial_capital:,.2f}")
        print(f"🎯 Symbols: {', '.join(self.symbols)}")
        print("=" * 60)
        
        await self.data_feed.start()
        self.data_feed.add_callback(self._on_market_data)
        
        self.running = True
        
        tasks = [
            asyncio.create_task(self._data_loop()),
            asyncio.create_task(self._trading_loop()),
            asyncio.create_task(self._monitoring_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n🛑 Trading system stopped by user")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading system"""
        self.running = False
        await self.data_feed.stop()
        
        # Print final results
        status = self.trading_engine.get_portfolio_status()
        print(f"\n📊 FINAL RESULTS:")
        print(f"💰 Balance: ${status['balance']}")
        print(f"📈 Total Return: {status['total_return']:.2f}%")
        print(f"🔄 Trades: {status['trades']}")
        print(f"📊 Signals Generated: {self.signal_count}")
    
    async def _data_loop(self):
        """Main data collection loop"""
        while self.running:
            try:
                # Get real live prices
                prices = await self.data_feed.get_live_prices(self.symbols)
                
                if prices:
                    logger.info(f"📡 Live prices updated: {len(prices)} symbols")
                else:
                    logger.warning("⚠️ No price data received")
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Data loop error: {e}")
                await asyncio.sleep(10)
    
    async def _trading_loop(self):
        """Main trading decision loop"""
        while self.running:
            try:
                for symbol in self.symbols:
                    # Generate trading signal
                    signal, confidence = self.technical_analysis.generate_trading_signal(symbol)
                    
                    if signal != "hold" and confidence > 0.7:
                        await self._process_trading_signal(symbol, signal, confidence)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(15)
    
    async def _monitoring_loop(self):
        """Monitor performance and positions"""
        while self.running:
            try:
                status = self.trading_engine.get_portfolio_status()
                
                print(f"\n📊 STATUS UPDATE:")
                print(f"   💰 Balance: ${status['balance']}")
                print(f"   📈 Total Return: {status['total_return']:.2f}%")
                print(f"   🔄 Active Positions: {status['positions']}")
                print(f"   📊 Completed Trades: {status['trades']}")
                print(f"   📡 Signals Generated: {self.signal_count}")
                
                # Check positions for stop loss / take profit
                await self._manage_positions()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _on_market_data(self, price_data: Dict):
        """Handle incoming real market data"""
        symbol = price_data['symbol']
        price = price_data['price']
        
        # Update technical analysis with real data
        self.technical_analysis.update_price(symbol, price)
        
        # Update position values if we have positions
        if symbol in self.trading_engine.positions:
            position = self.trading_engine.positions[symbol]
            position.current_price = price
            position.unrealized_pnl = (price - position.entry_price) * position.size
    
    async def _process_trading_signal(self, symbol: str, signal: str, confidence: float):
        """Process a trading signal"""
        
        # Avoid spam - only trade once per symbol per 5 minutes
        current_time = time.time()
        if symbol in self.last_signal_time:
            if current_time - self.last_signal_time[symbol] < 300:  # 5 minutes
                return
        
        self.last_signal_time[symbol] = current_time
        self.signal_count += 1
        
        # Get current price
        if symbol not in self.data_feed.price_cache:
            return
        
        current_price = self.data_feed.price_cache[symbol]['price']
        
        # Calculate position size (conservative)
        max_position_value = self.trading_engine.paper_balance * 0.1  # 10% max per trade
        position_size = max_position_value / current_price
        
        print(f"\n🎯 TRADING SIGNAL:")
        print(f"   Symbol: {symbol}")
        print(f"   Signal: {signal.upper()}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Price: ${current_price:.2f}")
        
        # Execute the trade
        if signal == "buy" and symbol not in self.trading_engine.positions:
            success = await self.trading_engine.execute_trade(
                symbol, "buy", position_size, current_price
            )
            if success:
                print(f"✅ Buy order executed for {symbol}")
        
        elif signal == "sell" and symbol in self.trading_engine.positions:
            success = await self.trading_engine.execute_trade(
                symbol, "sell", position_size, current_price
            )
            if success:
                print(f"✅ Sell order executed for {symbol}")
    
    async def _manage_positions(self):
        """Manage existing positions for stop loss / take profit"""
        for symbol, position in list(self.trading_engine.positions.items()):
            current_price = position.current_price
            
            # Check stop loss
            if current_price <= position.stop_loss:
                await self.trading_engine.execute_trade(
                    symbol, "sell", position.size, current_price
                )
                print(f"🛑 Stop loss triggered for {symbol}")
            
            # Check take profit
            elif current_price >= position.take_profit:
                await self.trading_engine.execute_trade(
                    symbol, "sell", position.size, current_price
                )
                print(f"🎯 Take profit triggered for {symbol}")

async def main():
    """Main function"""
    
    # Get trading mode from environment or command line
    mode = os.getenv('TRADING_MODE', 'paper')
    
    if len(os.sys.argv) > 1:
        if '--live' in os.sys.argv:
            mode = 'live'
        elif '--paper' in os.sys.argv:
            mode = 'paper'
        elif '--test' in os.sys.argv:
            # Quick test mode
            print("🧪 Running quick market data test...")
            data_feed = RealMarketDataFeed()
            await data_feed.start()
            
            symbols = ['BTCUSDT', 'ETHUSDT']
            prices = await data_feed.get_live_prices(symbols)
            
            if prices:
                print("✅ Market data test successful!")
                for symbol, data in prices.items():
                    print(f"   {symbol}: ${data['price']:.2f}")
            else:
                print("❌ Market data test failed!")
            
            await data_feed.stop()
            return
    
    if mode == 'live':
        print("🔴 LIVE TRADING MODE SELECTED")
        print("⚠️  WARNING: This will trade with real money!")
        response = input("Type 'CONFIRM' to proceed with live trading: ")
        if response != 'CONFIRM':
            print("❌ Live trading cancelled")
            return
    
    # Start the real trading system
    system = RealTradingSystem(mode)
    await system.start()

if __name__ == "__main__":
    import sys
    asyncio.run(main())