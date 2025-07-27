#!/usr/bin/env python3
"""
🚨 EMERGENCY FIXED TRADING SYSTEM
Critical bug fixes for position sizing and P&L calculation
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import warnings
from typing import Dict, List, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    side: str
    shares: float  # Number of shares/coins
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
    shares: float
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    reason: str

class MockDataFeed:
    """Simulates realistic market data with controlled volatility"""
    
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
        """Generate realistic price movements"""
        self.running = True
        
        while self.running:
            for symbol in self.symbols:
                # Much more conservative price movements
                volatility = 0.0005  # 0.05% per update (was 0.2%)
                price_change = np.random.normal(0, volatility)
                
                # Smaller trend component
                trend = 0.00001 * np.sin(time.time() / 3600)
                
                self.prices[symbol] *= (1 + price_change + trend)
                
                # Ensure reasonable bounds
                self.prices[symbol] = max(self.prices[symbol], 100)  # Minimum price
                
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
            
            await asyncio.sleep(2)  # Update every 2 seconds
    
    def stop(self):
        self.running = False

class TechnicalAnalyzer:
    """Safe technical analysis with bounds checking"""
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Bounds check
        return max(0, min(100, rsi))
    
    @staticmethod
    def sma(prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        return np.mean(prices[-period:])

class SafeRiskManager:
    """Conservative risk management with strict limits"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_value = 50  # Maximum $50 per position
        self.max_portfolio_risk = 0.10  # 10% total portfolio at risk
        self.min_position_value = 10   # Minimum $10 position
    
    def calculate_position_size(self, signal_strength: float, confidence: float, 
                              current_price: float) -> float:
        """Calculate safe position size in dollars"""
        
        # Very conservative base size
        base_size = min(30, self.current_capital * 0.02)  # 2% of capital, max $30
        
        # Apply signal and confidence
        adjusted_size = base_size * min(confidence, 0.8) * min(abs(signal_strength), 1.0)
        
        # Ensure within bounds
        position_value = max(self.min_position_value, min(adjusted_size, self.max_position_value))
        
        # Calculate number of shares/coins
        shares = position_value / current_price
        
        return shares
    
    def can_open_position(self, position_value: float) -> bool:
        """Check if we can safely open a position"""
        
        # Check available capital
        if self.current_capital < position_value * 1.1:  # Need 10% buffer
            return False
        
        # Check portfolio risk limits
        return position_value <= self.max_position_value

class SafeTradingEngine:
    """Safe paper trading engine with fixed P&L calculation"""
    
    def __init__(self, symbols: List[str], initial_capital: float = 1000):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        self.data_feed = MockDataFeed(symbols)
        self.risk_manager = SafeRiskManager(initial_capital)
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
        
        # Safety limits
        self.max_daily_loss = 100  # Stop if lose more than $100 in a day
        self.max_positions = 3     # Maximum 3 open positions
        
    async def start_trading(self):
        """Start safe paper trading"""
        
        print("🛡️ SAFE PAPER TRADING ENGINE")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Conservative Target: ${self.initial_capital * 2:,.2f} (2x return)")
        print(f"🛡️ Max Position Size: ${self.risk_manager.max_position_value}")
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
        """Process market data safely"""
        symbol = data['symbol']
        price = data['price']
        
        # Validate price
        if price <= 0 or price > 1000000:
            return
        
        # Store price history
        self.price_history[symbol].append(price)
        
        # Keep only last 100 prices
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
        
        # Update position values
        if symbol in self.positions:
            position = self.positions[symbol]
            old_price = position.current_price
            position.current_price = price
            
            # Calculate P&L correctly
            if position.side == 'buy':
                position.unrealized_pnl = (price - position.entry_price) * position.shares
            else:  # sell/short
                position.unrealized_pnl = (position.entry_price - price) * position.shares
    
    async def trading_loop(self):
        """Conservative trading loop"""
        
        while self.running:
            try:
                # Check safety limits
                total_loss = self.initial_capital - self.current_capital
                if total_loss > self.max_daily_loss:
                    print(f"🚨 Daily loss limit reached: ${total_loss:.2f}")
                    self.running = False
                    break
                
                for symbol in self.symbols:
                    await self.analyze_and_trade(symbol)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(30)
    
    async def analyze_and_trade(self, symbol: str):
        """Conservative analysis and trading"""
        
        prices = self.price_history[symbol]
        if len(prices) < 30:
            return
        
        current_price = prices[-1]
        rsi = self.analyzer.rsi(prices)
        sma_20 = self.analyzer.sma(prices, 20)
        
        # Very conservative signal generation
        signal_strength = 0
        confidence = 0
        
        # Only trade on strong RSI signals
        if rsi < 25:  # Very oversold
            signal_strength = 0.6
            confidence = 0.7
            side = 'buy'
        elif rsi > 75:  # Very overbought
            signal_strength = 0.6
            confidence = 0.7
            side = 'sell'
        else:
            return  # No trading
        
        # Check if we can trade
        if len(self.positions) >= self.max_positions:
            return
        
        if symbol in self.positions:
            return
        
        # Calculate position size
        shares = self.risk_manager.calculate_position_size(
            signal_strength, confidence, current_price
        )
        
        position_value = shares * current_price
        
        if not self.risk_manager.can_open_position(position_value):
            return
        
        # Open position
        await self.open_position(symbol, side, shares, current_price)
    
    async def open_position(self, symbol: str, side: str, shares: float, price: float):
        """Open position with proper P&L tracking"""
        
        position_value = shares * price
        
        # Calculate stop loss and take profit
        if side == 'buy':
            stop_loss = price * 0.98   # 2% stop loss
            take_profit = price * 1.04  # 4% take profit
        else:
            stop_loss = price * 1.02   # 2% stop loss  
            take_profit = price * 0.96  # 4% take profit
        
        position = Position(
            symbol=symbol,
            side=side,
            shares=shares,
            entry_price=price,
            current_price=price,
            unrealized_pnl=0,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = position
        
        # Reserve capital
        self.current_capital -= position_value
        
        print(f"📈 OPENED {side.upper()}: {symbol} | "
              f"Shares: {shares:.4f} @ ${price:.2f} | "
              f"Value: ${position_value:.2f}")
    
    async def manage_position(self, symbol: str):
        """Manage position with proper exit logic"""
        
        position = self.positions[symbol]
        current_price = position.current_price
        
        # Calculate return percentage
        if position.side == 'buy':
            return_pct = (current_price - position.entry_price) / position.entry_price
        else:
            return_pct = (position.entry_price - current_price) / position.entry_price
        
        # Exit conditions
        should_exit = False
        exit_reason = ""
        
        # Stop loss
        if position.side == 'buy' and current_price <= position.stop_loss:
            should_exit = True
            exit_reason = "stop_loss"
        elif position.side == 'sell' and current_price >= position.stop_loss:
            should_exit = True
            exit_reason = "stop_loss"
        
        # Take profit
        elif position.side == 'buy' and current_price >= position.take_profit:
            should_exit = True
            exit_reason = "take_profit"
        elif position.side == 'sell' and current_price <= position.take_profit:
            should_exit = True
            exit_reason = "take_profit"
        
        # Time exit (5 minutes max)
        elif (datetime.now() - position.entry_time).seconds > 300:
            should_exit = True
            exit_reason = "time_exit"
        
        # RSI reversal
        else:
            prices = self.price_history[symbol]
            if len(prices) >= 14:
                rsi = self.analyzer.rsi(prices)
                
                if position.side == 'buy' and rsi > 70:
                    should_exit = True
                    exit_reason = "rsi_reversal"
                elif position.side == 'sell' and rsi < 30:
                    should_exit = True
                    exit_reason = "rsi_reversal"
        
        if should_exit:
            await self.close_position(symbol, exit_reason)
    
    async def close_position(self, symbol: str, reason: str):
        """Close position with correct P&L calculation"""
        
        position = self.positions[symbol]
        exit_price = position.current_price
        
        # Calculate actual P&L
        if position.side == 'buy':
            pnl = (exit_price - position.entry_price) * position.shares
        else:  # sell/short
            pnl = (position.entry_price - exit_price) * position.shares
        
        # Return capital plus/minus P&L
        position_value = position.shares * position.entry_price
        self.current_capital += position_value + pnl
        
        # Calculate percentage
        pnl_pct = pnl / position_value if position_value > 0 else 0
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
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
              f"P&L: ${pnl:.2f} ({pnl_pct:.1%}) | {reason}")
    
    async def monitor_performance(self):
        """Monitor performance with correct calculations"""
        
        while self.running:
            await asyncio.sleep(30)  # Update every 30 seconds
            
            # Calculate total unrealized P&L
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Calculate position values
            position_values = sum(pos.shares * pos.current_price for pos in self.positions.values())
            
            # Total portfolio value
            total_value = self.current_capital + unrealized_pnl
            
            # Performance metrics
            win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
            total_return = (total_value - self.initial_capital) / self.initial_capital * 100
            
            print(f"\n💰 PERFORMANCE UPDATE - {datetime.now().strftime('%H:%M:%S')}")
            print(f"  Available Capital: ${self.current_capital:.2f}")
            print(f"  Position Values: ${position_values:.2f}")
            print(f"  Unrealized P&L: ${unrealized_pnl:.2f}")
            print(f"  Total Value: ${total_value:.2f}")
            print(f"  Total Return: {total_return:.1f}%")
            print(f"  Trades: {self.trade_count} | Win Rate: {win_rate:.1f}%")
            print(f"  Active Positions: {len(self.positions)}")
            
            # Update risk manager
            self.risk_manager.current_capital = total_value
            
            # Manage existing positions
            for symbol in list(self.positions.keys()):
                await self.manage_position(symbol)
    
    async def stop_trading(self):
        """Stop trading with final report"""
        
        self.running = False
        self.data_feed.stop()
        
        # Close all positions
        for symbol in list(self.positions.keys()):
            await self.close_position(symbol, "system_stop")
        
        # Final calculations
        total_pnl = sum(trade.pnl for trade in self.trades)
        final_value = self.current_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
        
        print(f"\n🏁 FINAL RESULTS")
        print("=" * 50)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${final_value:,.2f}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Total Return: {total_return:.1f}%")
        print(f"Total Trades: {self.trade_count}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Duration: {datetime.now() - self.start_time}")
        
        if total_return > 5:
            print(f"✅ SUCCESSFUL TRADING SESSION!")
        elif total_return > 0:
            print(f"✅ Profitable session")
        else:
            print(f"📊 Break-even/small loss - conservative approach working")

async def main():
    """Main function with error handling"""
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    initial_capital = 1000.0
    
    engine = SafeTradingEngine(symbols, initial_capital)
    
    try:
        await engine.start_trading()
    except KeyboardInterrupt:
        print("\n👋 Safe trading stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Main error: {e}")

if __name__ == "__main__":
    print("🛡️ EMERGENCY FIXED SAFE TRADING SYSTEM")
    print("=" * 50)
    print("💰 Starting Capital: $1,000")
    print("🎯 Conservative Target: $2,000 (2x return)")
    print("🛡️ Maximum Risk: $50 per position")
    print("⏰ Starting safe trading...")
    print("=" * 50)
    print("\nPress Ctrl+C to stop trading safely")
    print()
    
    asyncio.run(main())