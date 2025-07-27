#!/usr/bin/env python3
"""
🚀 FIXED LIVE PAPER TRADING SYSTEM
Standalone paper trading without external dependencies
"""

import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    leverage: float = 1.0

@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    reason: str

class MockDataFeed:
    """Simulates real-time market data"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.prices = {}
        self.running = False
        
        # Initialize with realistic prices
        base_prices = {
            'BTCUSDT': 65000,
            'ETHUSDT': 3200,
            'BNBUSDT': 580,
            'SOLUSDT': 180
        }
        
        for symbol in symbols:
            self.prices[symbol] = base_prices.get(symbol, 50000)
    
    async def start_feed(self, callback):
        """Start generating mock price data"""
        self.running = True
        
        while self.running:
            for symbol in self.symbols:
                # Generate realistic price movement
                volatility = 0.002  # 0.2% per update
                price_change = np.random.normal(0, volatility)
                
                # Add some trend bias
                trend = 0.0001 * np.sin(time.time() / 3600)  # Hourly cycle
                
                self.prices[symbol] *= (1 + price_change + trend)
                
                # Create market data
                price = self.prices[symbol]
                volume = np.random.exponential(100)
                spread = price * 0.0001  # 0.01% spread
                
                data = {
                    'symbol': symbol,
                    'price': price,
                    'bid': price - spread/2,
                    'ask': price + spread/2,
                    'volume': volume,
                    'timestamp': datetime.now()
                }
                
                await callback(data)
            
            await asyncio.sleep(1)  # Update every second
    
    def stop(self):
        self.running = False

class TechnicalAnalyzer:
    """Calculate technical indicators"""
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> float:
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
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def sma(prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        return np.mean(prices[-period:])
    
    @staticmethod
    def ema(prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        alpha = 2 / (period + 1)
        ema_val = prices[0]
        
        for price in prices[1:]:
            ema_val = alpha * price + (1 - alpha) * ema_val
        
        return ema_val

class RiskManager:
    """Advanced risk management"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_risk = 0.02  # 2% per trade
        self.max_portfolio_risk = 0.20  # 20% total
        self.max_leverage = 10
    
    def calculate_position_size(self, signal_strength: float, confidence: float, 
                              volatility: float) -> float:
        """Calculate optimal position size using Kelly criterion"""
        
        # Base Kelly calculation
        win_prob = 0.5 + confidence * 0.3  # Confidence boosts win probability
        avg_win = abs(signal_strength) * 2
        avg_loss = abs(signal_strength) * 1
        
        if avg_loss <= 0:
            return 0
        
        kelly = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        kelly = max(0, min(kelly, 0.25))  # Cap at 25%
        
        # Adjust for volatility
        vol_adjustment = 1 / (1 + volatility * 10)
        
        # Final position size
        position_size = self.current_capital * kelly * vol_adjustment * confidence
        
        # Apply risk limits
        max_position = self.current_capital * self.max_position_risk
        return min(position_size, max_position)

class PaperTradingEngine:
    """Complete paper trading engine"""
    
    def __init__(self, symbols: List[str], initial_capital: float = 1000):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        self.data_feed = MockDataFeed(symbols)
        self.risk_manager = RiskManager(initial_capital)
        self.analyzer = TechnicalAnalyzer()
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.price_history: Dict[str, List[float]] = {s: [] for s in symbols}
        self.running = False
        
        # Performance tracking
        self.start_time = datetime.now()
        self.trade_count = 0
        self.winning_trades = 0
        
    async def start_trading(self):
        """Start the paper trading system"""
        
        print("🚀 STARTING PAPER TRADING ENGINE")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Target: ${self.initial_capital * 10:,.2f} (10x return)")
        print("=" * 60)
        
        self.running = True
        
        # Start data feed
        feed_task = asyncio.create_task(
            self.data_feed.start_feed(self.process_market_data)
        )
        
        # Start trading loop
        trading_task = asyncio.create_task(self.trading_loop())
        
        # Start monitoring
        monitor_task = asyncio.create_task(self.monitor_performance())
        
        try:
            await asyncio.gather(feed_task, trading_task, monitor_task)
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        finally:
            await self.stop_trading()
    
    async def process_market_data(self, data: Dict):
        """Process incoming market data"""
        symbol = data['symbol']
        price = data['price']
        
        # Store price history
        self.price_history[symbol].append(price)
        
        # Keep only last 200 prices
        if len(self.price_history[symbol]) > 200:
            self.price_history[symbol] = self.price_history[symbol][-200:]
        
        # Update position values
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = price
            
            if position.side == 'buy':
                position.unrealized_pnl = (price - position.entry_price) * position.size * position.leverage
            else:
                position.unrealized_pnl = (position.entry_price - price) * position.size * position.leverage
    
    async def trading_loop(self):
        """Main trading logic loop"""
        
        while self.running:
            try:
                for symbol in self.symbols:
                    await self.analyze_and_trade(symbol)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)
    
    async def analyze_and_trade(self, symbol: str):
        """Analyze symbol and execute trades"""
        
        prices = self.price_history[symbol]
        if len(prices) < 50:
            return
        
        # Calculate technical indicators
        current_price = prices[-1]
        rsi = self.analyzer.rsi(prices)
        sma_20 = self.analyzer.sma(prices, 20)
        sma_50 = self.analyzer.sma(prices, 50)
        ema_12 = self.analyzer.ema(prices, 12)
        
        # Generate trading signals
        signals = []
        
        # RSI signals
        if rsi < 30:
            signals.append(('buy', 0.6, 'oversold'))
        elif rsi > 70:
            signals.append(('sell', 0.6, 'overbought'))
        
        # Moving average signals
        if current_price > sma_20 > sma_50:
            signals.append(('buy', 0.4, 'ma_trend'))
        elif current_price < sma_20 < sma_50:
            signals.append(('sell', 0.4, 'ma_trend'))
        
        # EMA crossover
        if ema_12 > sma_20 and current_price > ema_12:
            signals.append(('buy', 0.5, 'ema_cross'))
        elif ema_12 < sma_20 and current_price < ema_12:
            signals.append(('sell', 0.5, 'ema_cross'))
        
        # Mean reversion
        deviation = (current_price - sma_20) / sma_20
        if deviation < -0.02:  # 2% below MA
            signals.append(('buy', 0.3, 'mean_reversion'))
        elif deviation > 0.02:  # 2% above MA
            signals.append(('sell', 0.3, 'mean_reversion'))
        
        # Process signals
        if signals:
            await self.process_signals(symbol, signals, current_price)
        
        # Check existing positions
        if symbol in self.positions:
            await self.manage_position(symbol)
    
    async def process_signals(self, symbol: str, signals: List, current_price: float):
        """Process trading signals"""
        
        # Aggregate signals
        buy_strength = sum(strength for side, strength, reason in signals if side == 'buy')
        sell_strength = sum(strength for side, strength, reason in signals if side == 'sell')
        
        net_signal = buy_strength - sell_strength
        confidence = min(max(abs(net_signal), 0.1), 1.0)
        
        # Check if we should trade
        if abs(net_signal) < 0.3:  # Minimum signal threshold
            return
        
        # Don't trade if we already have a position
        if symbol in self.positions:
            return
        
        # Calculate volatility
        prices = self.price_history[symbol]
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            abs(net_signal), confidence, volatility
        )
        
        if position_size < 1:  # Minimum $1 position
            return
        
        # Execute trade
        side = 'buy' if net_signal > 0 else 'sell'
        leverage = min(self.risk_manager.max_leverage, confidence * 15)
        
        await self.open_position(symbol, side, position_size, current_price, leverage)
    
    async def open_position(self, symbol: str, side: str, size: float, 
                           price: float, leverage: float):
        """Open a new position"""
        
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            current_price=price,
            unrealized_pnl=0,
            entry_time=datetime.now(),
            leverage=leverage
        )
        
        self.positions[symbol] = position
        
        # Update capital (margin used)
        margin_used = size / leverage
        self.current_capital -= margin_used
        
        print(f"📈 OPENED {side.upper()}: {symbol} @ ${price:.2f} | "
              f"Size: ${size:.0f} | Leverage: {leverage:.1f}x")
    
    async def manage_position(self, symbol: str):
        """Manage existing position"""
        
        position = self.positions[symbol]
        current_price = position.current_price
        
        # Calculate returns
        if position.side == 'buy':
            return_pct = (current_price - position.entry_price) / position.entry_price
        else:
            return_pct = (position.entry_price - current_price) / position.entry_price
        
        leveraged_return = return_pct * position.leverage
        
        # Exit conditions
        should_exit = False
        exit_reason = ""
        
        # Profit taking (20% return)
        if leveraged_return > 0.20:
            should_exit = True
            exit_reason = "profit_target"
        
        # Stop loss (10% loss)
        elif leveraged_return < -0.10:
            should_exit = True
            exit_reason = "stop_loss"
        
        # Time-based exit (10 minutes)
        elif (datetime.now() - position.entry_time).seconds > 600:
            should_exit = True
            exit_reason = "time_exit"
        
        # Technical exit
        else:
            prices = self.price_history[symbol]
            if len(prices) >= 14:
                rsi = self.analyzer.rsi(prices)
                
                if position.side == 'buy' and rsi > 75:
                    should_exit = True
                    exit_reason = "rsi_exit"
                elif position.side == 'sell' and rsi < 25:
                    should_exit = True
                    exit_reason = "rsi_exit"
        
        if should_exit:
            await self.close_position(symbol, exit_reason)
    
    async def close_position(self, symbol: str, reason: str):
        """Close an existing position"""
        
        position = self.positions[symbol]
        exit_price = position.current_price
        
        # Calculate PnL
        if position.side == 'buy':
            return_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            return_pct = (position.entry_price - exit_price) / position.entry_price
        
        leveraged_return = return_pct * position.leverage
        pnl = position.size * leveraged_return
        
        # Update capital
        margin_used = position.size / position.leverage
        self.current_capital += margin_used + pnl
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            pnl=pnl,
            pnl_pct=leveraged_return,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            reason=reason
        )
        
        self.trades.append(trade)
        self.trade_count += 1
        
        if pnl > 0:
            self.winning_trades += 1
        
        # Remove position
        del self.positions[symbol]
        
        print(f"📊 CLOSED {position.side.upper()}: {symbol} @ ${exit_price:.2f} | "
              f"PnL: ${pnl:.2f} ({leveraged_return:.2%}) | {reason}")
    
    async def monitor_performance(self):
        """Monitor and display performance"""
        
        while self.running:
            await asyncio.sleep(30)  # Update every 30 seconds
            
            # Calculate metrics
            total_pnl = sum(trade.pnl for trade in self.trades)
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_value = self.current_capital + unrealized_pnl
            
            win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
            total_return = (total_value - self.initial_capital) / self.initial_capital * 100
            
            # Display status
            print(f"\n💰 PERFORMANCE UPDATE - {datetime.now().strftime('%H:%M:%S')}")
            print(f"  Capital: ${self.current_capital:.2f}")
            print(f"  Unrealized P&L: ${unrealized_pnl:.2f}")
            print(f"  Total Value: ${total_value:.2f}")
            print(f"  Total Return: {total_return:.1f}%")
            print(f"  Trades: {self.trade_count} | Win Rate: {win_rate:.1f}%")
            print(f"  Active Positions: {len(self.positions)}")
            
            # Check if target reached
            if total_value >= self.initial_capital * 10:
                print(f"\n🎉 TARGET REACHED! ${self.initial_capital:,.0f} → ${total_value:,.0f}")
                print("🏆 10x return achieved!")
                self.running = False
                break
    
    async def stop_trading(self):
        """Stop trading and show final results"""
        
        self.running = False
        self.data_feed.stop()
        
        # Close all positions
        for symbol in list(self.positions.keys()):
            await self.close_position(symbol, "system_stop")
        
        # Final results
        total_pnl = sum(trade.pnl for trade in self.trades)
        final_value = self.current_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
        
        print(f"\n🏁 FINAL RESULTS")
        print("=" * 50)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.1f}%")
        print(f"Total Trades: {self.trade_count}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Trading Duration: {datetime.now() - self.start_time}")
        
        if total_return > 0:
            print(f"✅ PROFITABLE TRADING SESSION!")
        else:
            print(f"⚠️ Unprofitable session - strategy needs adjustment")

async def main():
    """Main function"""
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    initial_capital = 1000.0
    
    engine = PaperTradingEngine(symbols, initial_capital)
    
    try:
        await engine.start_trading()
    except KeyboardInterrupt:
        print("\n👋 Paper trading stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Main error: {e}")

if __name__ == "__main__":
    print("🚀 FIXED LIVE PAPER TRADING SYSTEM")
    print("=" * 50)
    print("💰 Starting Capital: $1,000")
    print("🎯 Target: $10,000 (10x return)")
    print("⏰ Starting trading simulation...")
    print("=" * 50)
    print("\nPress Ctrl+C to stop trading at any time")
    print()
    
    asyncio.run(main())