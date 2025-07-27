use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub price: f64,
    pub quantity: f64,
    pub timestamp: u64,
    pub is_buyer_maker: bool,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct OrderBook {
    bids: BTreeMap<i64, OrderBookLevel>,
    asks: BTreeMap<i64, OrderBookLevel>,
    last_update: u64,
    symbol: String,
}

#[pymethods]
impl OrderBook {
    #[new]
    pub fn new(symbol: String) -> Self {
        OrderBook {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update: 0,
            symbol,
        }
    }

    pub fn update_bid(&mut self, price: f64, quantity: f64) {
        let timestamp = current_timestamp();
        let price_key = -(price * 1_000_000.0) as i64;
        
        match quantity {
            0.0 => { self.bids.remove(&price_key); },
            _ => { self.bids.insert(price_key, OrderBookLevel { price, quantity, timestamp }); }
        }
        self.last_update = timestamp;
    }

    pub fn update_ask(&mut self, price: f64, quantity: f64) {
        let timestamp = current_timestamp();
        let price_key = (price * 1_000_000.0) as i64;
        
        match quantity {
            0.0 => { self.asks.remove(&price_key); },
            _ => { self.asks.insert(price_key, OrderBookLevel { price, quantity, timestamp }); }
        }
        self.last_update = timestamp;
    }

    pub fn get_best_bid(&self) -> Option<(f64, f64)> {
        self.bids.first_key_value().map(|(_, level)| (level.price, level.quantity))
    }

    pub fn get_best_ask(&self) -> Option<(f64, f64)> {
        self.asks.first_key_value().map(|(_, level)| (level.price, level.quantity))
    }

    pub fn get_spread(&self) -> f64 {
        let (bid, ask) = (self.get_best_bid(), self.get_best_ask());
        match (bid, ask) {
            (Some((b, _)), Some((a, _))) => a - b,
            _ => 0.0,
        }
    }

    pub fn get_spread_percentage(&self) -> f64 {
        let (bid, ask) = (self.get_best_bid(), self.get_best_ask());
        match (bid, ask) {
            (Some((b, _)), Some((a, _))) => {
                let mid = (b + a) * 0.5;
                if mid > 0.0 { (a - b) / mid * 100.0 } else { 0.0 }
            }
            _ => 0.0,
        }
    }

    pub fn get_mid_price(&self) -> f64 {
        let (bid, ask) = (self.get_best_bid(), self.get_best_ask());
        match (bid, ask) {
            (Some((b, _)), Some((a, _))) => (b + a) * 0.5,
            _ => 0.0,
        }
    }

    pub fn get_volume_imbalance(&self, levels: usize) -> f64 {
        let bid_vol: f64 = self.bids.values().take(levels).map(|l| l.quantity).sum();
        let ask_vol: f64 = self.asks.values().take(levels).map(|l| l.quantity).sum();
        let total = bid_vol + ask_vol;
        if total > 0.0 { (bid_vol - ask_vol) / total } else { 0.0 }
    }

    pub fn get_depth_levels(&self, levels: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let bids = self.bids.values().take(levels).map(|l| (l.price, l.quantity)).collect();
        let asks = self.asks.values().take(levels).map(|l| (l.price, l.quantity)).collect();
        (bids, asks)
    }

    pub fn calculate_market_impact(&self, trade_size: f64, side: &str) -> f64 {
        let mut remaining = trade_size.abs();
        let (mut weighted_price, mut total_qty) = (0.0, 0.0);

        let iter: Box<dyn Iterator<Item = &OrderBookLevel>> = match side.to_lowercase().as_str() {
            "buy" => Box::new(self.asks.values()),
            _ => Box::new(self.bids.values()),
        };

        for level in iter {
            if remaining <= 0.0 { break; }
            let qty = remaining.min(level.quantity);
            weighted_price += level.price * qty;
            total_qty += qty;
            remaining -= qty;
        }

        if total_qty > 0.0 { weighted_price / total_qty } else { 0.0 }
    }

    pub fn get_liquidity_score(&self, levels: usize) -> f64 {
        let (bids, asks) = self.get_depth_levels(levels);
        let total_liq = bids.iter().map(|(_, q)| q).sum::<f64>() + asks.iter().map(|(_, q)| q).sum::<f64>();
        total_liq / (1.0 + self.get_spread_percentage())
    }
}

#[pyclass]
pub struct OrderBookAnalyzer {
    order_books: Arc<RwLock<HashMap<String, OrderBook>>>,
    trade_history: Arc<RwLock<HashMap<String, VecDeque<Trade>>>>,
    microstructure_cache: Arc<RwLock<HashMap<String, MicrostructureMetrics>>>,
}

#[derive(Debug, Clone)]
pub struct MicrostructureMetrics {
    pub effective_spread: f64,
    pub price_impact: f64,
    pub volume_imbalance: f64,
    pub trade_intensity: f64,
    pub volatility: f64,
    pub momentum: f64,
    pub mean_reversion_strength: f64,
    pub liquidity_score: f64,
    pub timestamp: u64,
}

#[pymethods]
impl OrderBookAnalyzer {
    #[new]
    pub fn new() -> Self {
        OrderBookAnalyzer {
            order_books: Arc::new(RwLock::new(HashMap::with_capacity(1024))),
            trade_history: Arc::new(RwLock::new(HashMap::with_capacity(1024))),
            microstructure_cache: Arc::new(RwLock::new(HashMap::with_capacity(1024))),
        }
    }

    pub fn update_order_book(&mut self, symbol: String, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) {
        let mut books = self.order_books.write().unwrap();
        let book = books.entry(symbol.clone()).or_insert_with(|| OrderBook::new(symbol));
        
        book.bids.clear();
        book.asks.clear();
        
        bids.into_iter().for_each(|(p, q)| { book.update_bid(p, q); });
        asks.into_iter().for_each(|(p, q)| { book.update_ask(p, q); });
    }

    pub fn add_trade(&mut self, symbol: String, price: f64, quantity: f64, is_buyer_maker: bool) {
        let trade = Trade { price, quantity, timestamp: current_timestamp(), is_buyer_maker };
        let mut trades = self.trade_history.write().unwrap();
        let symbol_trades = trades.entry(symbol.clone()).or_insert_with(|| VecDeque::with_capacity(1000));
        
        symbol_trades.push_back(trade);
        if symbol_trades.len() > 1000 { symbol_trades.pop_front(); }
        drop(trades);
        self.update_microstructure_metrics(symbol);
    }

    pub fn get_microstructure_features(&self, symbol: &str) -> Vec<f64> {
        self.microstructure_cache.read().unwrap().get(symbol)
            .map(|m| vec![m.effective_spread, m.price_impact, m.volume_imbalance, m.trade_intensity, 
                         m.volatility, m.momentum, m.mean_reversion_strength, m.liquidity_score])
            .unwrap_or_else(|| vec![0.0; 8])
    }

    pub fn detect_anomalies(&self, symbol: &str) -> Vec<String> {
        let mut anomalies = Vec::with_capacity(8);
        let books = self.order_books.read().unwrap();
        
        if let Some(book) = books.get(symbol) {
            if book.get_spread_percentage() > 1.0 { anomalies.push("wide_spread".to_string()); }
            if book.get_liquidity_score(10) < 1000.0 { anomalies.push("low_liquidity".to_string()); }
            if book.get_volume_imbalance(5).abs() > 0.8 { anomalies.push("high_imbalance".to_string()); }
        }
        
        let trades = self.trade_history.read().unwrap();
        if let Some(symbol_trades) = trades.get(symbol) {
            if symbol_trades.len() > 10 {
                let recent: Vec<_> = symbol_trades.iter().rev().take(10).collect();
                let prices: Vec<f64> = recent.iter().map(|t| t.price).collect();
                let volatility = calculate_volatility(&prices);
                
                if volatility > 0.05 { anomalies.push("high_volatility".to_string()); }
                
                let avg_size = recent.iter().map(|t| t.quantity).sum::<f64>() / recent.len() as f64;
                if recent[0].quantity > avg_size * 5.0 { anomalies.push("large_trade".to_string()); }
            }
        }
        anomalies
    }
    
    opportunities.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    
    Ok(opportunities)
}

#[pyfunction]
pub fn parallel_orderbook_processing(
    orderbook_data: Vec<(String, Vec<(f64, f64)>, Vec<(f64, f64)>)>
) -> PyResult<Vec<(String, f64, f64, f64, f64, f64)>> {
    let results = orderbook_data.into_par_iter().map(|(symbol, bids, asks)| {
        let mut book = OrderBook::new(symbol.clone());
        bids.into_iter().for_each(|(p, q)| book.update_bid(p, q));
        asks.into_iter().for_each(|(p, q)| book.update_ask(p, q));
        
        (symbol, book.get_mid_price(), book.get_spread(), book.get_spread_percentage(), 
         book.get_volume_imbalance(5), book.get_liquidity_score(10))
    }).collect();
    
    Ok(results)
}

#[pymodule]
fn orderbook_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OrderBook>()?;
    m.add_class::<OrderBookAnalyzer>()?;
    m.add_function(wrap_pyfunction!(fast_arbitrage_detection, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_orderbook_processing, m)?)?;
    Ok(())
}
        
    pub fn calculate_optimal_execution_price(&self, symbol: &str, side: &str, target_volume: f64) -> f64 {
        self.order_books.read().unwrap().get(symbol)
            .map(|book| book.calculate_market_impact(target_volume, side))
            .unwrap_or(0.0)
    }

    pub fn get_order_flow_imbalance(&self, symbol: &str, window_seconds: u64) -> f64 {
        let trades = self.trade_history.read().unwrap();
        trades.get(symbol).map(|symbol_trades| {
            let cutoff = current_timestamp().saturating_sub(window_seconds * 1000);
            let recent: Vec<_> = symbol_trades.iter().filter(|t| t.timestamp >= cutoff).collect();
            
            if recent.is_empty() { return 0.0; }
            
            let (buy_vol, sell_vol) = recent.iter().fold((0.0, 0.0), |(b, s), t| {
                if t.is_buyer_maker { (b, s + t.quantity) } else { (b + t.quantity, s) }
            });
            
            let total = buy_vol + sell_vol;
            if total > 0.0 { (buy_vol - sell_vol) / total } else { 0.0 }
        }).unwrap_or(0.0)
    }

    pub fn predict_short_term_direction(&self, symbol: &str) -> f64 {
        let features = self.get_microstructure_features(symbol);
        if features.len() < 8 { return 0.0; }
        
        let order_flow = self.get_order_flow_imbalance(symbol, 60);
        let signal = features[5] * 0.3 + features[2] * 0.3 + order_flow * 0.25 - features[6] * 0.15;
        signal.clamp(-1.0, 1.0)
    }

    fn update_microstructure_metrics(&self, symbol: String) {
        let books = self.order_books.read().unwrap();
        let trades = self.trade_history.read().unwrap();
        
        if let (Some(book), Some(symbol_trades)) = (books.get(&symbol), trades.get(&symbol)) {
            if symbol_trades.len() < 5 { return; }
            
            let recent: Vec<_> = symbol_trades.iter().rev().take(20).collect();
            let prices: Vec<f64> = recent.iter().map(|t| t.price).collect();
            
            let metrics = MicrostructureMetrics {
                effective_spread: book.get_spread_percentage(),
                price_impact: self.calculate_price_impact(&recent),
                volume_imbalance: book.get_volume_imbalance(5),
                trade_intensity: recent.len() as f64 / 60.0,
                volatility: calculate_volatility(&prices),
                momentum: calculate_momentum(&prices),
                mean_reversion_strength: calculate_mean_reversion(&prices),
                liquidity_score: book.get_liquidity_score(10),
                timestamp: current_timestamp(),
            };
            
            self.microstructure_cache.write().unwrap().insert(symbol, metrics);
        }
    }

    fn calculate_price_impact(&self, trades: &[&Trade]) -> f64 {
        if trades.len() < 2 { return 0.0; }
        
        let total_volume: f64 = trades.iter().map(|t| t.quantity).sum();
        let price_changes: f64 = trades.windows(2).map(|w| (w[1].price - w[0].price).abs()).sum();
        
        if total_volume > 0.0 { price_changes / total_volume } else { 0.0 }
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
}

fn calculate_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 { return 0.0; }
    
    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    if returns.is_empty() { return 0.0; }
    
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    variance.sqrt()
}

fn calculate_momentum(prices: &[f64]) -> f64 {
    if prices.len() < 5 { return 0.0; }
    (prices[prices.len() - 1] / prices[0]) - 1.0
}

fn calculate_mean_reversion(prices: &[f64]) -> f64 {
    if prices.len() < 3 { return 0.0; }
    
    let mean_price = prices.iter().sum::<f64>() / prices.len() as f64;
    let current_price = prices[prices.len() - 1];
    ((current_price - mean_price) / mean_price).abs()
}

#[pyfunction]
pub fn fast_arbitrage_detection(order_books: Vec<(String, Vec<(f64, f64)>, Vec<(f64, f64)>)>, 
                               min_profit_threshold: f64) 
    -> PyResult<Vec<(String, String, f64, f64, f64)>> {
    let mut opportunities = Vec::with_capacity(1024);
    let mut symbol_books: HashMap<String, Vec<(String, Vec<(f64, f64)>, Vec<(f64, f64)>)>> = HashMap::with_capacity(256);
    
    order_books.into_iter().for_each(|(exchange, bids, asks)| {
        symbol_books.entry("BTCUSDT".to_string()).or_insert_with(Vec::new).push((exchange, bids, asks));
    });
    
    for (symbol, books) in symbol_books {
        for i in 0..books.len() {
            for j in (i + 1)..books.len() {
                let (ref ex1, ref bids1, ref asks1) = books[i];
                let (ref ex2, ref bids2, ref asks2) = books[j];
                
                if let (Some((bid1, _)), Some((ask2, _))) = (bids1.first(), asks2.first()) {
                    let profit = (bid1 - ask2) / ask2;
                    if profit > min_profit_threshold {
                        opportunities.push((symbol.clone(), format!("{}→{}", ex2, ex1), *ask2, *bid1, profit));
                    }
                }
                
                if let (Some((bid2, _)), Some((ask1, _))) = (bids2.first(), asks1.first()) {
                    let profit = (bid2 - ask1) / ask1;
                    if profit > min_profit_threshold {
                        opportunities.push((symbol.clone(), format!("{}→{}", ex1, ex2), *ask1, *bid2, profit));
                    }
                }
            }
        }
    }
    
    opportunities.sort_unstable_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    Ok(opportunities)
}