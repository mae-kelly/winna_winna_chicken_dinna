import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Union, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import aioredis
import asyncpg
from datetime import datetime, timedelta
import time
import logging
import threading
from collections import deque, defaultdict
import zlib
import gzip
from functools import lru_cache, wraps
import uvloop
import orjson
from numba import njit, prange
import cython
from scipy import stats
from sklearn.preprocessing import RobustScaler
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

@dataclass(slots=True, frozen=True)
class MarketData:
    symbol: str
    timestamp: float
    price: float
    volume: float
    bid: float
    ask: float
    exchange: str
    order_book: Optional[Dict] = field(default=None, compare=False)
    trades: Optional[List] = field(default=None, compare=False)

    @property
    def spread_bps(self) -> float:
        return 10000 * (self.ask - self.bid) / ((self.ask + self.bid) / 2) if self.ask > 0 and self.bid > 0 else float('inf')

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2 if self.ask > 0 and self.bid > 0 else self.price

class WebSocketManager:
    __slots__ = ('connections', 'callbacks', 'running', 'reconnect_delays', '_connection_pool', '_message_cache', '_compression_ctx', '_rate_limiters')

    def __init__(self):
        self.connections = {}
        self.callbacks = {}
        self.running = False
        self.reconnect_delays = defaultdict(lambda: 1)
        self._connection_pool = {}
        self._message_cache = {}
        self._compression_ctx = {'gzip': gzip.GzipFile, 'zlib': zlib.decompressobj()}
        self._rate_limiters = defaultdict(lambda: {'last_call': 0, 'calls': 0})

    async def connect_exchange(self, exchange: str, symbols: List[str], callback: Callable):
        self.callbacks[exchange] = callback

        exchange_handlers = {
            'binance': self._connect_binance,
            'coinbase': self._connect_coinbase,
            'kraken': self._connect_kraken,
            'bybit': self._connect_bybit,
            'okx': self._connect_okx
        }

        if handler := exchange_handlers.get(exchange):
            await handler(symbols)

    async def _connect_binance(self, symbols: List[str]):
        symbol_map = {s.replace('/', '').lower(): s for s in symbols}
        streams = [f"{sym}@ticker/{sym}@depth20@100ms/{sym}@trade" for sym in symbol_map.keys()]

        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        await self._maintain_connection('binance', url, self._handle_binance_message, compression='none')

    async def _connect_coinbase(self, symbols: List[str]):
        url = "wss://advanced-trade-ws.coinbase.com"
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": symbols,
            "channels": [{"name": "ticker_batch", "product_ids": symbols},
                        {"name": "level2_batch", "product_ids": symbols},
                        {"name": "market_trades", "product_ids": symbols}],
            "jwt": None
        }
        await self._maintain_connection('coinbase', url, self._handle_coinbase_message, subscribe_msg)

    async def _connect_kraken(self, symbols: List[str]):
        url = "wss://ws.kraken.com/v2"
        subscribe_msg = {
            "method": "subscribe",
            "params": {
                "channel": "ticker",
                "symbol": symbols,
                "snapshot": False
            },
            "req_id": int(time.time() * 1000)
        }
        await self._maintain_connection('kraken', url, self._handle_kraken_message, subscribe_msg)

    async def _connect_bybit(self, symbols: List[str]):
        url = "wss://stream.bybit.com/v5/public/linear"
        symbol_clean = [s.replace('/', '') for s in symbols]
        topics = [f"tickers.{s}" for s in symbol_clean] + [f"orderbook.50.{s}" for s in symbol_clean]

        subscribe_msg = {"op": "subscribe", "args": topics, "req_id": f"bybit_{int(time.time())}"}
        await self._maintain_connection('bybit', url, self._handle_bybit_message, subscribe_msg)

    async def _connect_okx(self, symbols: List[str]):
        url = "wss://ws.okx.com:8443/ws/v5/public"
        args = []
        for symbol in symbols:
            symbol_okx = symbol.replace('/', '-')
            args.extend([
                {"channel": "tickers", "instId": symbol_okx},
                {"channel": "books5", "instId": symbol_okx},
                {"channel": "trades", "instId": symbol_okx}
            ])

        subscribe_msg = {"op": "subscribe", "args": args, "id": f"okx_{int(time.time())}"}
        await self._maintain_connection('okx', url, self._handle_okx_message, subscribe_msg)

    async def _maintain_connection(self, exchange: str, url: str, handler: Callable, subscribe_msg: Optional[Dict] = None, compression: str = 'gzip'):
        while self.running:
            try:
                headers = {"User-Agent": f"AdvancedTrader-{exchange}/1.0"}
                extra_headers = {"Accept-Encoding": "gzip, deflate"} if compression != 'none' else {}

                async with websockets.connect(
                    url,
                    ping_interval=15,
                    ping_timeout=8,
                    max_size=2**23,
                    compression=None,
                    extra_headers={**headers, **extra_headers}
                ) as websocket:
                    self.connections[exchange] = websocket

                    if subscribe_msg:
                        await websocket.send(orjson.dumps(subscribe_msg).decode())

                    logger.info(f"🔗 {exchange.upper()} connected")
                    self.reconnect_delays[exchange] = 1

                    async for message in websocket:
                        try:
                            if isinstance(message, bytes):
                                message = self._decompress_message(message, compression)

                            data = orjson.loads(message) if isinstance(message, (str, bytes)) else message
                            await handler(data)

                        except orjson.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"⚠️  {exchange} message error: {e}")

            except Exception as e:
                delay = min(self.reconnect_delays[exchange] * 1.5, 60)
                logger.error(f"❌ {exchange} connection failed: {e}. Retry in {delay:.1f}s")
                await asyncio.sleep(delay)
                self.reconnect_delays[exchange] = delay

    def _decompress_message(self, message: bytes, compression: str) -> str:
        try:
            if compression == 'gzip':
                return gzip.decompress(message).decode('utf-8')
            elif compression == 'zlib':
                return zlib.decompress(message).decode('utf-8')
            return message.decode('utf-8')
        except:
            return message.decode('utf-8', errors='ignore')

    async def _handle_binance_message(self, data):
        if stream := data.get('stream'):
            payload = data['data']

            if '@ticker' in stream:
                symbol = stream.split('@')[0].upper()
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=payload['E'] / 1000,
                    price=float(payload['c']),
                    volume=float(payload['v']),
                    bid=float(payload['b']),
                    ask=float(payload['a']),
                    exchange='binance'
                )
                await self.callbacks['binance'](market_data)

    async def _handle_coinbase_message(self, data):
        if data.get('channel') == 'ticker_batch':
            for event in data.get('events', []):
                if tickers := event.get('tickers'):
                    for ticker in tickers:
                        market_data = MarketData(
                            symbol=ticker['product_id'],
                            timestamp=time.time(),
                            price=float(ticker['price']),
                            volume=float(ticker.get('volume_24_h', 0)),
                            bid=float(ticker['best_bid']),
                            ask=float(ticker['best_ask']),
                            exchange='coinbase'
                        )
                        await self.callbacks['coinbase'](market_data)

    async def _handle_kraken_message(self, data):
        if data.get('channel') == 'ticker' and (ticker_data := data.get('data')):
            for tick in ticker_data:
                market_data = MarketData(
                    symbol=tick['symbol'],
                    timestamp=time.time(),
                    price=float(tick['last']),
                    volume=float(tick['volume']),
                    bid=float(tick['bid']),
                    ask=float(tick['ask']),
                    exchange='kraken'
                )
                await self.callbacks['kraken'](market_data)

    async def _handle_bybit_message(self, data):
        if data.get('topic', '').startswith('tickers') and (ticker_data := data.get('data')):
            market_data = MarketData(
                symbol=ticker_data['symbol'],
                timestamp=int(ticker_data['ts']) / 1000,
                price=float(ticker_data['lastPrice']),
                volume=float(ticker_data['volume24h']),
                bid=float(ticker_data['bid1Price']),
                ask=float(ticker_data['ask1Price']),
                exchange='bybit'
            )
            await self.callbacks['bybit'](market_data)

    async def _handle_okx_message(self, data):
        if data.get('arg', {}).get('channel') == 'tickers' and (ticker_data := data.get('data')):
            for tick in ticker_data:
                market_data = MarketData(
                    symbol=tick['instId'].replace('-', '/'),
                    timestamp=int(tick['ts']) / 1000,
                    price=float(tick['last']),
                    volume=float(tick['vol24h']),
                    bid=float(tick['bidPx']),
                    ask=float(tick['askPx']),
                    exchange='okx'
                )
                await self.callbacks['okx'](market_data)

class HighFrequencyDataBuffer:
    __slots__ = ('data', 'max_size', 'lock', '_feature_cache', '_numpy_cache', '_stats_cache', '_scaler')

    def __init__(self, max_size: int = 100000):
        self.data = {}
        self.max_size = max_size
        self.lock = threading.RLock()
        self._feature_cache = {}
        self._numpy_cache = {}
        self._stats_cache = {}
        self._scaler = RobustScaler()

    def add_tick(self, market_data: MarketData):
        with self.lock:
            symbol = market_data.symbol
            if symbol not in self.data:
                self.data[symbol] = deque(maxlen=self.max_size)

            tick_data = {
                'timestamp': market_data.timestamp,
                'price': market_data.price,
                'volume': market_data.volume,
                'bid': market_data.bid,
                'ask': market_data.ask,
                'exchange': market_data.exchange,
                'spread': market_data.ask - market_data.bid,
                'mid_price': market_data.mid_price,
                'spread_bps': market_data.spread_bps,
                'tick_direction': np.sign(market_data.price - (self.data[symbol][-1]['price'] if self.data[symbol] else market_data.price))
            }

            self.data[symbol].append(tick_data)
            self._invalidate_caches(symbol)

    def _invalidate_caches(self, symbol: str):
        cache_keys_to_remove = [k for k in self._feature_cache.keys() if k.startswith(symbol)]
        for key in cache_keys_to_remove:
            self._feature_cache.pop(key, None)
            self._numpy_cache.pop(key, None)
            self._stats_cache.pop(key, None)

    @lru_cache(maxsize=1000)
    def get_recent_data(self, symbol: str, seconds: int = 60) -> pd.DataFrame:
        with self.lock:
            if symbol not in self.data:
                return pd.DataFrame()

            cutoff_time = time.time() - seconds
            recent_data = [tick for tick in self.data[symbol] if tick['timestamp'] >= cutoff_time]

            if not recent_data:
                return pd.DataFrame()

            df = pd.DataFrame(recent_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp').reset_index(drop=True)

            if len(df) > 1:
                df['price_change'] = df['price'].diff()
                df['volume_weighted_price'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
                df['volatility_proxy'] = df['price_change'].rolling(min(10, len(df)), min_periods=1).std()
                df['momentum'] = df['price'].pct_change(min(5, len(df)-1)).fillna(0)

            return df

    def get_microstructure_features(self, symbol: str, window_seconds: int = 10) -> Dict:
        cache_key = f"{symbol}_{window_seconds}_{int(time.time()/5)*5}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        df = self.get_recent_data(symbol, window_seconds)

        if len(df) < 2:
            return {}

        features = self._compute_advanced_features(df, window_seconds)
        self._feature_cache[cache_key] = features

        return features

    @njit(parallel=True, fastmath=True)
    def _compute_price_features(prices: np.ndarray, volumes: np.ndarray) -> Tuple[float, float, float, float]:
        n = len(prices)
        if n < 2:
            return 0.0, 0.0, 0.0, 0.0

        price_changes = np.diff(prices)
        weighted_changes = price_changes * volumes[1:]

        velocity = np.mean(price_changes)
        acceleration = np.mean(np.diff(price_changes)) if n > 2 else 0.0
        weighted_velocity = np.sum(weighted_changes) / np.sum(volumes[1:]) if np.sum(volumes[1:]) > 0 else 0.0
        volatility = np.std(price_changes) * np.sqrt(252 * 24 * 3600)

        return velocity, acceleration, weighted_velocity, volatility

    def _compute_advanced_features(self, df: pd.DataFrame, window_seconds: int) -> Dict:
        try:
            prices = df['price'].values
            volumes = df['volume'].values
            spreads = df['spread'].values

            if len(prices) < 2:
                return {}

            price_changes = np.diff(prices)

            velocity, acceleration, weighted_velocity, volatility = self._compute_price_features(prices, volumes)

            tick_directions = np.sign(price_changes)
            trade_intensity = len(df) / window_seconds

            volume_profile = np.histogram(prices, bins=min(10, len(prices)//2), weights=volumes)[0]
            volume_concentration = np.max(volume_profile) / np.sum(volume_profile) if np.sum(volume_profile) > 0 else 0

            microstructure_noise = np.std(price_changes) / np.mean(np.abs(price_changes)) if np.mean(np.abs(price_changes)) > 0 else 0

            order_flow_imbalance = np.mean(df['bid'] - df['ask']) / np.mean(df['mid_price']) if len(df) > 0 else 0

            realized_variance = np.sum(price_changes**2)
            garman_klass_volatility = np.log(df['ask']/df['bid']).var() if len(df) > 1 else 0

            features = {
                'price_velocity': float(velocity),
                'price_acceleration': float(acceleration),
                'weighted_velocity': float(weighted_velocity),
                'volume_imbalance': float(order_flow_imbalance),
                'spread_volatility': float(np.std(spreads)),
                'tick_direction_bias': float(np.mean(tick_directions)),
                'trade_intensity': float(trade_intensity),
                'price_impact': float(np.sum(np.abs(price_changes)) / np.sum(volumes) if np.sum(volumes) > 0 else 0),
                'volatility_estimate': float(volatility),
                'volume_concentration': float(volume_concentration),
                'microstructure_noise': float(microstructure_noise),
                'realized_variance': float(realized_variance),
                'garman_klass_vol': float(garman_klass_volatility),
                'effective_spread': float(np.mean(spreads) / np.mean(prices) if len(prices) > 0 else 0),
                'price_efficiency': float(1 / (1 + microstructure_noise)),
                'momentum_score': float(np.corrcoef(range(len(prices)), prices)[0,1] if len(prices) > 1 else 0)
            }

            return features

        except Exception as e:
            logger.error(f"Feature computation error: {e}")
            return {}

class CrossExchangeArbitrageDetector:
    __slots__ = ('min_profit_threshold', 'price_cache', '_opportunity_cache', '_ml_predictor', '_risk_metrics')

    def __init__(self, min_profit_threshold: float = 0.001):
        self.min_profit_threshold = min_profit_threshold
        self.price_cache = {}
        self._opportunity_cache = {}
        self._ml_predictor = None
        self._risk_metrics = defaultdict(list)

    def update_price(self, market_data: MarketData):
        key = f"{market_data.symbol}_{market_data.exchange}"

        self.price_cache[key] = {
            'timestamp': market_data.timestamp,
            'bid': market_data.bid,
            'ask': market_data.ask,
            'price': market_data.price,
            'volume': market_data.volume,
            'spread_bps': market_data.spread_bps,
            'quality_score': self._assess_price_quality(market_data)
        }

        self._update_risk_metrics(market_data)

    def _assess_price_quality(self, market_data: MarketData) -> float:
        quality_factors = [
            1.0 if market_data.spread_bps < 50 else 0.5,
            1.0 if market_data.volume > 1000 else market_data.volume / 1000,
            1.0 if time.time() - market_data.timestamp < 1 else 0.1
        ]
        return np.mean(quality_factors)

    def _update_risk_metrics(self, market_data: MarketData):
        symbol = market_data.symbol
        self._risk_metrics[symbol].append({
            'timestamp': market_data.timestamp,
            'price': market_data.price,
            'spread': market_data.spread_bps,
            'exchange': market_data.exchange
        })

        if len(self._risk_metrics[symbol]) > 1000:
            self._risk_metrics[symbol] = self._risk_metrics[symbol][-500:]

    def detect_opportunities(self, symbol: str) -> List[Dict]:
        cache_key = f"{symbol}_{int(time.time())}"
        if cache_key in self._opportunity_cache:
            return self._opportunity_cache[cache_key]

        opportunities = []
        valid_exchanges = []
        current_time = time.time()

        for key, data in self.price_cache.items():
            if key.startswith(f"{symbol}_") and current_time - data['timestamp'] < 3:
                if data['quality_score'] > 0.5:
                    exchange = key.split('_', 1)[1]
                    valid_exchanges.append((exchange, data))

        if len(valid_exchanges) < 2:
            return []

        for i, (ex1, data1) in enumerate(valid_exchanges):
            for ex2, data2 in valid_exchanges[i+1:]:
                opportunities.extend(self._evaluate_arbitrage_pair(symbol, ex1, data1, ex2, data2, current_time))

        opportunities = sorted(opportunities, key=lambda x: x['risk_adjusted_profit'], reverse=True)[:5]
        self._opportunity_cache[cache_key] = opportunities

        return opportunities

    def _evaluate_arbitrage_pair(self, symbol: str, ex1: str, data1: Dict, ex2: str, data2: Dict, timestamp: float) -> List[Dict]:
        opportunities = []

        risk_penalty1 = self._calculate_exchange_risk(ex1, symbol)
        risk_penalty2 = self._calculate_exchange_risk(ex2, symbol)

        for buy_ex, buy_data, sell_ex, sell_data, risk_buy, risk_sell in [
            (ex2, data2, ex1, data1, risk_penalty2, risk_penalty1),
            (ex1, data1, ex2, data2, risk_penalty1, risk_penalty2)
        ]:
            if sell_data['bid'] > buy_data['ask'] * (1 + self.min_profit_threshold):
                gross_profit = (sell_data['bid'] - buy_data['ask']) / buy_data['ask']

                execution_cost = (buy_data['spread_bps'] + sell_data['spread_bps']) / 10000
                slippage_cost = max(0.0001, gross_profit * 0.1)
                total_risk = (risk_buy + risk_sell) / 2

                net_profit = gross_profit - execution_cost - slippage_cost
                risk_adjusted_profit = net_profit / (1 + total_risk)

                if risk_adjusted_profit > self.min_profit_threshold:
                    confidence = min(buy_data['quality_score'], sell_data['quality_score'])

                    opportunity = {
                        'symbol': symbol,
                        'buy_exchange': buy_ex,
                        'sell_exchange': sell_ex,
                        'buy_price': buy_data['ask'],
                        'sell_price': sell_data['bid'],
                        'profit_pct': gross_profit,
                        'net_profit_pct': net_profit,
                        'risk_adjusted_profit': risk_adjusted_profit,
                        'execution_cost': execution_cost,
                        'confidence_score': confidence,
                        'timestamp': timestamp,
                        'expiry_estimate': timestamp + min(5, 1/max(0.001, gross_profit))
                    }
                    opportunities.append(opportunity)

        return opportunities

    def _calculate_exchange_risk(self, exchange: str, symbol: str) -> float:
        if symbol not in self._risk_metrics:
            return 0.5

        recent_data = [d for d in self._risk_metrics[symbol]
                      if d['exchange'] == exchange and time.time() - d['timestamp'] < 300]

        if len(recent_data) < 5:
            return 0.3

        prices = [d['price'] for d in recent_data]
        spreads = [d['spread'] for d in recent_data]

        price_volatility = np.std(prices) / np.mean(prices) if prices else 0.1
        avg_spread = np.mean(spreads) if spreads else 50

        risk_score = min(1.0, (price_volatility * 10 + avg_spread / 100) / 2)
        return risk_score

class RealTimeDataEngine:
    __slots__ = ('symbols', 'ws_manager', 'data_buffer', 'arbitrage_detector', 'callbacks', 'exchanges',
                 '_performance_monitor', '_redis_client', '_db_pool', '_ml_engine', '_health_monitor')

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.ws_manager = WebSocketManager()
        self.data_buffer = HighFrequencyDataBuffer()
        self.arbitrage_detector = CrossExchangeArbitrageDetector()
        self.callbacks = []
        self.exchanges = ['binance', 'coinbase', 'kraken', 'bybit', 'okx']
        self._performance_monitor = defaultdict(lambda: {'count': 0, 'latency': [], 'errors': 0})
        self._redis_client = None
        self._db_pool = None
        self._ml_engine = None
        self._health_monitor = {'start_time': time.time(), 'message_count': 0, 'error_count': 0}

    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)

    async def start(self):
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

        self.ws_manager.running = True
        self._health_monitor['start_time'] = time.time()

        connection_tasks = [
            asyncio.create_task(
                self.ws_manager.connect_exchange(exchange, self.symbols, self._handle_market_data),
                name=f"connect_{exchange}"
            ) for exchange in self.exchanges
        ]

        monitoring_task = asyncio.create_task(self._monitor_performance(), name="performance_monitor")

        all_tasks = connection_tasks + [monitoring_task]

        try:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Engine startup error: {e}")

    async def _monitor_performance(self):
        while self.ws_manager.running:
            await asyncio.sleep(30)

            uptime = time.time() - self._health_monitor['start_time']
            msg_rate = self._health_monitor['message_count'] / uptime if uptime > 0 else 0
            error_rate = self._health_monitor['error_count'] / uptime if uptime > 0 else 0

            logger.info(f"📊 Performance: {msg_rate:.1f} msg/s, {error_rate:.3f} err/s, {uptime:.0f}s uptime")

            for exchange, metrics in self._performance_monitor.items():
                if metrics['latency']:
                    avg_latency = np.mean(metrics['latency'][-100:])
                    logger.info(f"   {exchange}: {avg_latency:.1f}ms avg latency, {metrics['count']} messages")

    async def _handle_market_data(self, market_data: MarketData):
        start_time = time.perf_counter()

        try:
            self.data_buffer.add_tick(market_data)
            self.arbitrage_detector.update_price(market_data)

            self._health_monitor['message_count'] += 1

            callback_tasks = [
                asyncio.create_task(callback(market_data))
                for callback in self.callbacks
            ]

            if callback_tasks:
                await asyncio.gather(*callback_tasks, return_exceptions=True)

        except Exception as e:
            self._health_monitor['error_count'] += 1
            logger.error(f"Market data handling error: {e}")

        finally:
            processing_time = (time.perf_counter() - start_time) * 1000
            self._performance_monitor[market_data.exchange]['latency'].append(processing_time)
            self._performance_monitor[market_data.exchange]['count'] += 1

    def get_latest_features(self, symbol: str) -> Dict:
        try:
            microstructure = self.data_buffer.get_microstructure_features(symbol, 10)
            arbitrage_ops = self.arbitrage_detector.detect_opportunities(symbol)
            data_quality = self._assess_data_quality(symbol)

            market_regime = self._detect_market_regime(symbol)
            liquidity_metrics = self._calculate_liquidity_metrics(symbol)

            return {
                'microstructure': microstructure,
                'arbitrage_opportunities': arbitrage_ops,
                'data_quality': data_quality,
                'market_regime': market_regime,
                'liquidity_metrics': liquidity_metrics,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Feature extraction error for {symbol}: {e}")
            return {'error': str(e), 'timestamp': time.time()}

    def _detect_market_regime(self, symbol: str) -> Dict:
        df = self.data_buffer.get_recent_data(symbol, 300)

        if len(df) < 10:
            return {'regime': 'unknown', 'confidence': 0}

        price_changes = df['price'].pct_change().dropna()
        volatility = price_changes.std()
        trend_strength = abs(price_changes.mean()) / volatility if volatility > 0 else 0

        if volatility > price_changes.std() * 2:
            regime = 'high_volatility'
        elif trend_strength > 0.1:
            regime = 'trending'
        elif volatility < price_changes.std() * 0.5:
            regime = 'low_volatility'
        else:
            regime = 'normal'

        confidence = min(1.0, len(df) / 100)

        return {
            'regime': regime,
            'confidence': confidence,
            'volatility_percentile': stats.percentileofscore(price_changes, volatility),
            'trend_strength': trend_strength
        }

    def _calculate_liquidity_metrics(self, symbol: str) -> Dict:
        df = self.data_buffer.get_recent_data(symbol, 60)

        if len(df) < 5:
            return {}

        spreads = df['spread_bps']
        volumes = df['volume']

        return {
            'avg_spread_bps': float(spreads.mean()),
            'spread_volatility': float(spreads.std()),
            'volume_weighted_spread': float(np.average(spreads, weights=volumes)) if volumes.sum() > 0 else 0,
            'liquidity_score': float(1 / (1 + spreads.mean() / 100)),
            'market_depth_proxy': float(volumes.mean()),
            'price_impact_estimate': float(spreads.mean() / volumes.mean() if volumes.mean() > 0 else float('inf'))
        }

    def _assess_data_quality(self, symbol: str) -> Dict:
        df = self.data_buffer.get_recent_data(symbol, 60)

        if len(df) == 0:
            return {'score': 0, 'issues': ['no_data'], 'recommendations': ['check_connections']}

        issues = []
        recommendations = []

        data_freshness = time.time() - df['timestamp'].max().timestamp()
        if data_freshness > 10:
            issues.append('stale_data')
            recommendations.append('check_websocket_connections')

        if len(df) < 20:
            issues.append('insufficient_data')
            recommendations.append('wait_for_more_data')

        spread_threshold = df['price'].mean() * 0.005
        if df['spread'].mean() > spread_threshold:
            issues.append('wide_spreads')
            recommendations.append('use_limit_orders')

        price_jumps = abs(df['price'].pct_change()) > 0.03
        if price_jumps.sum() > len(df) * 0.05:
            issues.append('price_instability')
            recommendations.append('increase_risk_controls')

        exchange_coverage = df['exchange'].nunique()
        if exchange_coverage < 3:
            issues.append('limited_exchange_coverage')
            recommendations.append('add_more_exchanges')

        outlier_threshold = 3
        z_scores = np.abs(stats.zscore(df['price']))
        if (z_scores > outlier_threshold).sum() > len(df) * 0.02:
            issues.append('price_outliers')
            recommendations.append('implement_outlier_filtering')

        base_score = 1.0
        score_penalties = {
            'no_data': 1.0, 'stale_data': 0.3, 'insufficient_data': 0.2,
            'wide_spreads': 0.15, 'price_instability': 0.25,
            'limited_exchange_coverage': 0.1, 'price_outliers': 0.2
        }

        for issue in issues:
            base_score -= score_penalties.get(issue, 0.1)

        score = max(0, base_score)

        return {
            'score': score,
            'issues': issues,
            'recommendations': recommendations,
            'data_points': len(df),
            'exchange_coverage': exchange_coverage,
            'freshness_seconds': data_freshness
        }

    async def stop(self):
        logger.info("🛑 Shutting down Real-Time Data Engine")
        self.ws_manager.running = False

        close_tasks = []
        for exchange, connection in self.ws_manager.connections.items():
            if connection:
                close_tasks.append(asyncio.create_task(connection.close(), name=f"close_{exchange}"))

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        if self._redis_client:
            await self._redis_client.close()

        if self._db_pool:
            await self._db_pool.close()

        logger.info("✅ Engine shutdown complete")

async def main():
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']

    engine = RealTimeDataEngine(symbols)

    async def data_callback(market_data: MarketData):
        print(f"🚀 {market_data.exchange.upper()}: {market_data.symbol} @ ${market_data.price:,.6f} | Spread: {market_data.spread_bps:.1f}bps")

    engine.add_callback(data_callback)

    try:
        await engine.start()
    except KeyboardInterrupt:
        logger.info("👋 Received shutdown signal")
    finally:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
