#!/usr/bin/env python3

import asyncio
import aiohttp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
import uvloop
import orjson
from scipy import stats
from datetime import datetime, timedelta
import cupy as cp
import cudf
import threading
from threading import Lock
import hashlib
import redis.asyncio as redis

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    print(f"🚀 GPU Scanner Active: {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print("⚠️ Running on CPU")

@dataclass
class TokenData:
    symbol: str
    address: str
    price: float
    price_1m: float
    price_3m: float
    price_5m: float
    volume_24h: float
    volume_1h: float
    volume_15m: float
    liquidity: float
    market_cap: float
    price_change_1m: float
    price_change_3m: float
    price_change_5m: float
    volume_surge: float
    timestamp: float
    pair_address: str = ""
    chain: str = "ethereum"

@dataclass
class BreakoutCandidate:
    token_data: TokenData
    breakout_confidence: float
    entropy_score: float
    momentum_vector: np.ndarray
    liquidity_score: float
    safety_score: float
    final_score: float
    timestamp: float

class TransformerBreakoutModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=512, num_heads=16, num_layers=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.breakout_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.momentum_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.Tanh()
        )
        
        self.entropy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        x = self.input_projection(x)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        pooled = torch.mean(x, dim=1)
        
        breakout_conf = self.breakout_head(pooled)
        momentum_vec = self.momentum_head(pooled)
        entropy_score = self.entropy_head(pooled)
        
        return {
            'breakout_confidence': breakout_conf.squeeze(-1),
            'momentum_vector': momentum_vec,
            'entropy_score': entropy_score.squeeze(-1)
        }

class DexScreenerAPI:
    def __init__(self):
        self.session = None
        self.base_url = "https://api.dexscreener.com/latest"
        self.rate_limit = asyncio.Semaphore(50)
        self.request_count = 0
        self.last_reset = time.time()
        
    async def start(self):
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=50,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=10, connect=3)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'HyperMomentumScanner/1.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
        )
    
    async def stop(self):
        if self.session:
            await self.session.close()
    
    async def get_trending_tokens(self, limit: int = 500) -> List[Dict]:
        async with self.rate_limit:
            try:
                url = f"{self.base_url}/dex/pairs/ethereum"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('pairs', [])[:limit]
                    else:
                        logger.warning(f"DexScreener API error: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"DexScreener request failed: {e}")
                return []
    
    async def get_token_details(self, addresses: List[str]) -> List[Dict]:
        if not addresses:
            return []
        
        chunks = [addresses[i:i+30] for i in range(0, len(addresses), 30)]
        results = []
        
        for chunk in chunks:
            async with self.rate_limit:
                try:
                    address_list = ','.join(chunk)
                    url = f"{self.base_url}/dex/tokens/{address_list}"
                    
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'pairs' in data:
                                results.extend(data['pairs'])
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Token details request failed: {e}")
        
        return results
    
    async def search_tokens(self, query: str = "") -> List[Dict]:
        async with self.rate_limit:
            try:
                url = f"{self.base_url}/dex/search?q={query}"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('pairs', [])
                    return []
            except Exception as e:
                logger.error(f"Token search failed: {e}")
                return []

class TokenProcessor:
    def __init__(self):
        self.price_history = defaultdict(lambda: deque(maxlen=100))
        self.volume_history = defaultdict(lambda: deque(maxlen=50))
        self.last_seen = {}
        self.blacklist = set()
        self.lock = Lock()
        
    def process_token(self, token_data: Dict) -> Optional[TokenData]:
        try:
            base_token = token_data.get('baseToken', {})
            quote_token = token_data.get('quoteToken', {})
            
            if not base_token or not quote_token:
                return None
            
            symbol = base_token.get('symbol', '')
            address = base_token.get('address', '')
            
            if not symbol or not address or address in self.blacklist:
                return None
            
            price_usd = float(token_data.get('priceUsd', 0))
            if price_usd <= 0:
                return None
            
            volume_24h = float(token_data.get('volume', {}).get('h24', 0))
            volume_1h = float(token_data.get('volume', {}).get('h1', 0))
            volume_5m = float(token_data.get('volume', {}).get('m5', 0))
            
            liquidity_usd = float(token_data.get('liquidity', {}).get('usd', 0))
            market_cap = float(token_data.get('marketCap', 0))
            
            price_change_5m = float(token_data.get('priceChange', {}).get('m5', 0))
            price_change_1h = float(token_data.get('priceChange', {}).get('h1', 0))
            price_change_24h = float(token_data.get('priceChange', {}).get('h24', 0))
            
            current_time = time.time()
            
            with self.lock:
                self.price_history[address].append((current_time, price_usd))
                self.volume_history[address].append((current_time, volume_5m))
                self.last_seen[address] = current_time
            
            price_1m, price_3m, price_5m = self._calculate_historical_prices(address, current_time)
            
            if price_1m <= 0 or price_3m <= 0 or price_5m <= 0:
                price_1m = price_3m = price_5m = price_usd
            
            change_1m = ((price_usd - price_1m) / price_1m) * 100 if price_1m > 0 else 0
            change_3m = ((price_usd - price_3m) / price_3m) * 100 if price_3m > 0 else 0
            change_5m = ((price_usd - price_5m) / price_5m) * 100 if price_5m > 0 else price_change_5m
            
            volume_baseline = self._calculate_volume_baseline(address)
            volume_surge = ((volume_5m - volume_baseline) / volume_baseline) * 100 if volume_baseline > 0 else 0
            
            return TokenData(
                symbol=symbol,
                address=address,
                price=price_usd,
                price_1m=price_1m,
                price_3m=price_3m,
                price_5m=price_5m,
                volume_24h=volume_24h,
                volume_1h=volume_1h,
                volume_15m=volume_5m * 3,
                liquidity=liquidity_usd,
                market_cap=market_cap,
                price_change_1m=change_1m,
                price_change_3m=change_3m,
                price_change_5m=change_5m,
                volume_surge=volume_surge,
                timestamp=current_time,
                pair_address=token_data.get('pairAddress', ''),
                chain=token_data.get('chainId', 'ethereum')
            )
            
        except Exception as e:
            logger.error(f"Token processing error: {e}")
            return None
    
    def _calculate_historical_prices(self, address: str, current_time: float) -> Tuple[float, float, float]:
        history = self.price_history[address]
        if len(history) < 2:
            return 0, 0, 0
        
        price_1m = self._interpolate_price(history, current_time - 60)
        price_3m = self._interpolate_price(history, current_time - 180)
        price_5m = self._interpolate_price(history, current_time - 300)
        
        return price_1m, price_3m, price_5m
    
    def _interpolate_price(self, history: deque, target_time: float) -> float:
        if not history:
            return 0
        
        history_list = list(history)
        
        if len(history_list) < 2:
            return history_list[0][1] if history_list else 0
        
        before = None
        after = None
        
        for timestamp, price in history_list:
            if timestamp <= target_time:
                before = (timestamp, price)
            elif timestamp > target_time and after is None:
                after = (timestamp, price)
                break
        
        if before and after:
            time_diff = after[0] - before[0]
            if time_diff > 0:
                weight = (target_time - before[0]) / time_diff
                return before[1] + weight * (after[1] - before[1])
        
        if before:
            return before[1]
        elif after:
            return after[1]
        else:
            return history_list[-1][1]
    
    def _calculate_volume_baseline(self, address: str) -> float:
        history = self.volume_history[address]
        if len(history) < 5:
            return 1.0
        
        volumes = [vol for _, vol in list(history)]
        return np.median(volumes) if volumes else 1.0

class HoneypotDetector:
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300
        
    async def check_token_safety(self, address: str, session: aiohttp.ClientSession) -> float:
        if address in self.cache:
            timestamp, score = self.cache[address]
            if time.time() - timestamp < self.cache_duration:
                return score
        
        try:
            honeypot_score = await self._check_honeypot(address, session)
            blacklist_score = await self._check_blacklist(address, session)
            liquidity_score = await self._check_liquidity_lock(address, session)
            
            final_score = min(honeypot_score, blacklist_score, liquidity_score)
            
            self.cache[address] = (time.time(), final_score)
            return final_score
            
        except Exception as e:
            logger.error(f"Safety check failed for {address}: {e}")
            return 0.5
    
    async def _check_honeypot(self, address: str, session: aiohttp.ClientSession) -> float:
        try:
            url = f"https://api.honeypot.is/v2/IsHoneypot?address={address}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('IsHoneypot', False):
                        return 0.0
                    return 0.9
                return 0.7
        except:
            return 0.7
    
    async def _check_blacklist(self, address: str, session: aiohttp.ClientSession) -> float:
        known_scam_patterns = [
            'test', 'fake', 'scam', 'honeypot', 'rug', 'phishing'
        ]
        
        address_lower = address.lower()
        for pattern in known_scam_patterns:
            if pattern in address_lower:
                return 0.0
        
        return 0.95
    
    async def _check_liquidity_lock(self, address: str, session: aiohttp.ClientSession) -> float:
        return 0.8

class GPUFeatureExtractor:
    def __init__(self):
        self.device = device
        
    def extract_features(self, token_data: TokenData) -> torch.Tensor:
        features = [
            token_data.price_change_1m / 100,
            token_data.price_change_3m / 100,
            token_data.price_change_5m / 100,
            np.log1p(token_data.volume_surge) / 10,
            np.log1p(token_data.liquidity) / 20,
            np.log1p(token_data.market_cap) / 25,
            np.log1p(token_data.volume_24h) / 15,
            np.log1p(token_data.volume_1h) / 12,
            token_data.price / max(token_data.price_5m, 1e-8),
            token_data.volume_1h / max(token_data.volume_24h, 1e-8),
            np.tanh(token_data.price_change_1m / 5),
            np.tanh(token_data.price_change_3m / 8),
            np.tanh(token_data.price_change_5m / 10),
            min(token_data.volume_surge / 1000, 1.0),
            min(token_data.liquidity / 1000000, 1.0),
            min(token_data.market_cap / 10000000, 1.0)
        ]
        
        momentum_features = self._calculate_momentum_features(token_data)
        acceleration_features = self._calculate_acceleration_features(token_data)
        volatility_features = self._calculate_volatility_features(token_data)
        
        all_features = features + momentum_features + acceleration_features + volatility_features
        
        while len(all_features) < 32:
            all_features.append(0.0)
        
        return torch.tensor(all_features[:32], dtype=torch.float32, device=self.device)
    
    def _calculate_momentum_features(self, token_data: TokenData) -> List[float]:
        price_momentum = (token_data.price - token_data.price_5m) / max(token_data.price_5m, 1e-8)
        volume_momentum = token_data.volume_surge / 100
        
        return [
            np.tanh(price_momentum),
            np.tanh(volume_momentum),
            price_momentum * volume_momentum,
            np.sign(price_momentum) * np.sqrt(abs(price_momentum))
        ]
    
    def _calculate_acceleration_features(self, token_data: TokenData) -> List[float]:
        change_1m = token_data.price_change_1m / 100
        change_3m = token_data.price_change_3m / 100
        change_5m = token_data.price_change_5m / 100
        
        accel_1_3 = change_1m - change_3m
        accel_3_5 = change_3m - change_5m
        
        return [
            np.tanh(accel_1_3 * 10),
            np.tanh(accel_3_5 * 10),
            accel_1_3 * accel_3_5,
            np.sign(accel_1_3) * np.sign(accel_3_5)
        ]
    
    def _calculate_volatility_features(self, token_data: TokenData) -> List[float]:
        price_changes = [
            token_data.price_change_1m,
            token_data.price_change_3m,
            token_data.price_change_5m
        ]
        
        volatility = np.std(price_changes)
        mean_change = np.mean(price_changes)
        
        return [
            volatility / 100,
            mean_change / 100,
            volatility / max(abs(mean_change), 1e-8),
            max(price_changes) - min(price_changes)
        ]

class EntropyCalculator:
    @staticmethod
    def calculate_entropy_decay(momentum_vector: np.ndarray, price_changes: List[float]) -> float:
        if len(price_changes) < 3:
            return 0.5
        
        try:
            price_array = np.array(price_changes)
            
            hist, _ = np.histogram(price_array, bins=5, density=True)
            hist = hist[hist > 0]
            
            if len(hist) < 2:
                return 0.3
            
            shannon_entropy = -np.sum(hist * np.log2(hist))
            max_entropy = np.log2(len(hist))
            normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
            
            momentum_magnitude = np.linalg.norm(momentum_vector)
            momentum_entropy = 1.0 / (1.0 + momentum_magnitude)
            
            acceleration = np.diff(price_array)
            if len(acceleration) > 1:
                accel_entropy = np.std(acceleration) / (np.mean(np.abs(acceleration)) + 1e-8)
                accel_entropy = min(accel_entropy, 2.0) / 2.0
            else:
                accel_entropy = 0.5
            
            final_entropy = (normalized_entropy * 0.4 + 
                           momentum_entropy * 0.3 + 
                           accel_entropy * 0.3)
            
            return max(0.0, min(1.0, final_entropy))
            
        except Exception as e:
            logger.error(f"Entropy calculation error: {e}")
            return 0.5

class HyperMomentumScanner:
    def __init__(self):
        self.dex_api = DexScreenerAPI()
        self.token_processor = TokenProcessor()
        self.honeypot_detector = HoneypotDetector()
        self.feature_extractor = GPUFeatureExtractor()
        self.entropy_calculator = EntropyCalculator()
        
        self.model = TransformerBreakoutModel().to(device)
        self.model.eval()
        
        self.candidate_queue = asyncio.Queue(maxsize=1000)
        self.processed_tokens = set()
        self.scan_count = 0
        self.candidates_found = 0
        
        self.min_liquidity = 50000
        self.min_volume_24h = 10000
        self.min_market_cap = 100000
        self.max_market_cap = 100000000
        
        self.breakout_min = 8.0
        self.breakout_max = 13.0
        self.confidence_threshold = 0.85
        self.entropy_threshold = 0.4
        
        self.running = False
        
        torch.jit.script(self.model)
    
    async def start(self):
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        await self.dex_api.start()
        self.running = True
        
        logger.info("🚀 Hyper-Momentum Scanner started")
        
        tasks = [
            asyncio.create_task(self._scan_loop()),
            asyncio.create_task(self._process_candidates()),
            asyncio.create_task(self._monitor_performance())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Scanner shutdown requested")
        finally:
            await self.stop()
    
    async def stop(self):
        self.running = False
        await self.dex_api.stop()
    
    async def _scan_loop(self):
        while self.running:
            try:
                start_time = time.time()
                
                trending_tokens = await self.dex_api.get_trending_tokens(2000)
                
                if not trending_tokens:
                    await asyncio.sleep(5)
                    continue
                
                batch_size = 50
                token_batches = [trending_tokens[i:i+batch_size] 
                               for i in range(0, len(trending_tokens), batch_size)]
                
                tasks = []
                for batch in token_batches:
                    task = asyncio.create_task(self._process_token_batch(batch))
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                scan_duration = time.time() - start_time
                self.scan_count += 1
                
                if self.scan_count % 10 == 0:
                    logger.info(f"Scan #{self.scan_count} completed in {scan_duration:.2f}s, "
                              f"{len(trending_tokens)} tokens processed")
                
                await asyncio.sleep(max(0.5, 2.0 - scan_duration))
                
            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                await asyncio.sleep(5)
    
    async def _process_token_batch(self, token_batch: List[Dict]):
        for token_raw in token_batch:
            try:
                token_data = self.token_processor.process_token(token_raw)
                if not token_data:
                    continue
                
                if not self._passes_initial_filters(token_data):
                    continue
                
                if not self._in_breakout_zone(token_data):
                    continue
                
                safety_score = await self.honeypot_detector.check_token_safety(
                    token_data.address, self.dex_api.session
                )
                
                if safety_score < 0.7:
                    continue
                
                features = self.feature_extractor.extract_features(token_data)
                features = features.unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    model_output = self.model(features)
                    
                    breakout_confidence = model_output['breakout_confidence'].item()
                    momentum_vector = model_output['momentum_vector'].cpu().numpy().flatten()
                    entropy_score = model_output['entropy_score'].item()
                
                if breakout_confidence < self.confidence_threshold:
                    continue
                
                price_changes = [
                    token_data.price_change_1m,
                    token_data.price_change_3m,
                    token_data.price_change_5m
                ]
                
                entropy_decay = self.entropy_calculator.calculate_entropy_decay(
                    momentum_vector, price_changes
                )
                
                if entropy_decay < self.entropy_threshold:
                    continue
                
                liquidity_score = min(token_data.liquidity / 1000000, 1.0)
                
                final_score = (breakout_confidence * 0.4 + 
                             entropy_score * 0.3 + 
                             safety_score * 0.2 + 
                             liquidity_score * 0.1)
                
                candidate = BreakoutCandidate(
                    token_data=token_data,
                    breakout_confidence=breakout_confidence,
                    entropy_score=entropy_score,
                    momentum_vector=momentum_vector,
                    liquidity_score=liquidity_score,
                    safety_score=safety_score,
                    final_score=final_score,
                    timestamp=time.time()
                )
                
                try:
                    await self.candidate_queue.put(candidate)
                    self.candidates_found += 1
                except asyncio.QueueFull:
                    pass
                
            except Exception as e:
                logger.error(f"Token batch processing error: {e}")
    
    def _passes_initial_filters(self, token_data: TokenData) -> bool:
        if token_data.liquidity < self.min_liquidity:
            return False
        
        if token_data.volume_24h < self.min_volume_24h:
            return False
        
        if token_data.market_cap < self.min_market_cap or token_data.market_cap > self.max_market_cap:
            return False
        
        if token_data.address in self.processed_tokens:
            return False
        
        return True
    
    def _in_breakout_zone(self, token_data: TokenData) -> bool:
        changes = [
            token_data.price_change_1m,
            token_data.price_change_3m,
            token_data.price_change_5m
        ]
        
        for change in changes:
            if self.breakout_min <= change <= self.breakout_max:
                return True
        
        return False
    
    async def _process_candidates(self):
        while self.running:
            try:
                candidate = await asyncio.wait_for(
                    self.candidate_queue.get(), timeout=1.0
                )
                
                output_data = {
                    'symbol': candidate.token_data.symbol,
                    'address': candidate.token_data.address,
                    'price': candidate.token_data.price,
                    'price_change_1m': candidate.token_data.price_change_1m,
                    'price_change_3m': candidate.token_data.price_change_3m,
                    'price_change_5m': candidate.token_data.price_change_5m,
                    'volume_surge': candidate.token_data.volume_surge,
                    'breakout_confidence': candidate.breakout_confidence,
                    'entropy_score': candidate.entropy_score,
                    'liquidity': candidate.token_data.liquidity,
                    'market_cap': candidate.token_data.market_cap,
                    'safety_score': candidate.safety_score,
                    'final_score': candidate.final_score,
                    'timestamp': candidate.timestamp,
                    'momentum_vector': candidate.momentum_vector.tolist(),
                    'pair_address': candidate.token_data.pair_address,
                    'chain': candidate.token_data.chain
                }
                
                print(orjson.dumps(output_data).decode())
                
                self.processed_tokens.add(candidate.token_data.address)
                
                if len(self.processed_tokens) > 10000:
                    oldest_tokens = list(self.processed_tokens)[:5000]
                    for token in oldest_tokens:
                        self.processed_tokens.discard(token)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Candidate processing error: {e}")
    
    async def _monitor_performance(self):
        start_time = time.time()
        
        while self.running:
            await asyncio.sleep(30)
            
            uptime = time.time() - start_time
            scan_rate = self.scan_count / (uptime / 60) if uptime > 0 else 0
            candidate_rate = self.candidates_found / (uptime / 60) if uptime > 0 else 0
            
            gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            logger.info(f"📊 Performance: {scan_rate:.1f} scans/min, "
                       f"{candidate_rate:.2f} candidates/min, "
                       f"GPU: {gpu_memory:.1f}GB, "
                       f"Queue: {self.candidate_queue.qsize()}")

async def main():
    scanner = HyperMomentumScanner()
    
    try:
        await scanner.start()
    except KeyboardInterrupt:
        logger.info("🛑 Scanner stopped by user")
    except Exception as e:
        logger.error(f"Scanner error: {e}")
    finally:
        await scanner.stop()

if __name__ == "__main__":
    asyncio.run(main())