use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use std::sync::Arc;
use std::collections::VecDeque;

#[pyclass]
pub struct FastMathEngine {
    feature_cache: Arc<std::sync::Mutex<VecDeque<Vec<f64>>>>,
    correlation_matrix: Arc<std::sync::Mutex<Vec<Vec<f64>>>>,
}

#[pymethods]
impl FastMathEngine {
    #[new]
    pub fn new() -> Self {
        FastMathEngine {
            feature_cache: Arc::new(std::sync::Mutex::new(VecDeque::with_capacity(16384))),
            correlation_matrix: Arc::new(std::sync::Mutex::new(vec![vec![0.0; 256]; 256])),
        }
    }

    pub fn fast_rsi(&self, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<f64> {
        let prices = prices.as_slice()?;
        if prices.len() <= period { return Ok(50.0); }

        let (mut gain_sum, mut loss_sum) = prices.windows(2)
            .rev().take(period)
            .fold((0.0, 0.0), |(g, l), w| {
                let delta = w[1] - w[0];
                if delta > 0.0 { (g + delta, l) } else { (g, l - delta) }
            });

        if loss_sum == 0.0 { return Ok(100.0); }
        let rs = gain_sum / loss_sum;
        Ok(100.0 - 100.0 / (1.0 + rs))
    }

    pub fn fast_bollinger_bands(&self, prices: PyReadonlyArray1<f64>, period: usize, std_dev: f64) 
        -> PyResult<(f64, f64, f64)> {
        let prices = prices.as_slice()?;
        if prices.len() < period {
            let last = *prices.last().unwrap_or(&0.0);
            return Ok((last, last, last));
        }

        let recent = &prices[prices.len() - period..];
        let (sum, sum_sq) = recent.iter().fold((0.0, 0.0), |(s, sq), &p| (s + p, sq + p * p));
        let mean = sum / period as f64;
        let variance = sum_sq / period as f64 - mean * mean;
        let std = variance.sqrt() * std_dev;

        Ok((mean + std, mean, mean - std))
    }

    pub fn fast_ema(&self, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<f64> {
        let prices = prices.as_slice()?;
        if prices.is_empty() { return Ok(0.0); }

        let alpha = 2.0 / (period as f64 + 1.0);
        let beta = 1.0 - alpha;
        
        Ok(prices.iter().skip(1).fold(prices[0], |ema, &price| alpha * price + beta * ema))
    }

    pub fn fast_macd(&self, prices: PyReadonlyArray1<f64>, fast_period: usize, 
                     slow_period: usize, signal_period: usize) 
        -> PyResult<(f64, f64, f64)> {
        let prices = prices.as_slice()?;
        
        if prices.len() < slow_period.max(fast_period) { return Ok((0.0, 0.0, 0.0)); }

        let fast_alpha = 2.0 / (fast_period as f64 + 1.0);
        let slow_alpha = 2.0 / (slow_period as f64 + 1.0);
        let signal_alpha = 2.0 / (signal_period as f64 + 1.0);
        
        let (fast_ema, slow_ema) = prices.iter().skip(1)
            .fold((prices[0], prices[0]), |(f, s), &p| {
                (fast_alpha * p + (1.0 - fast_alpha) * f, slow_alpha * p + (1.0 - slow_alpha) * s)
            });
        
        let macd = fast_ema - slow_ema;
        let signal = signal_alpha * macd + (1.0 - signal_alpha) * macd;
        
        Ok((macd, signal, macd - signal))
    }

    pub fn fast_stochastic(&self, highs: PyReadonlyArray1<f64>, lows: PyReadonlyArray1<f64>, 
                          closes: PyReadonlyArray1<f64>, period: usize) 
        -> PyResult<f64> {
        let (highs, lows, closes) = (highs.as_slice()?, lows.as_slice()?, closes.as_slice()?);

        if highs.len() < period { return Ok(50.0); }

        let end = highs.len();
        let (highest, lowest) = (end.saturating_sub(period)..end)
            .fold((f64::NEG_INFINITY, f64::INFINITY), |(h, l), i| {
                (h.max(highs[i]), l.min(lows[i]))
            });

        if highest == lowest { return Ok(50.0); }
        Ok(((closes[end - 1] - lowest) / (highest - lowest)) * 100.0)
    }

    pub fn fast_atr(&self, highs: PyReadonlyArray1<f64>, lows: PyReadonlyArray1<f64>, 
                    closes: PyReadonlyArray1<f64>, period: usize) 
        -> PyResult<f64> {
        let (highs, lows, closes) = (highs.as_slice()?, lows.as_slice()?, closes.as_slice()?);

        if highs.len() < 2 { return Ok(0.0); }

        let tr_sum: f64 = (1..highs.len()).rev().take(period.min(highs.len() - 1))
            .map(|i| (highs[i] - lows[i]).max((highs[i] - closes[i-1]).abs()).max((lows[i] - closes[i-1]).abs()))
            .sum();

        Ok(tr_sum / period.min(highs.len() - 1) as f64)
    }

    pub fn fast_correlation(&self, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) 
        -> PyResult<f64> {
        let (x, y) = (x.as_slice()?, y.as_slice()?);

        if x.len() != y.len() || x.len() < 2 { return Ok(0.0); }

        let n = x.len() as f64;
        let (sum_x, sum_y, sum_xy, sum_x2, sum_y2) = x.iter().zip(y.iter())
            .fold((0.0, 0.0, 0.0, 0.0, 0.0), |(sx, sy, sxy, sx2, sy2), (&xi, &yi)| {
                (sx + xi, sy + yi, sxy + xi * yi, sx2 + xi * xi, sy2 + yi * yi)
            });

        let num = n * sum_xy - sum_x * sum_y;
        let den = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if den == 0.0 { Ok(0.0) } else { Ok(num / den) }
    }

    pub fn fast_volatility(&self, prices: PyReadonlyArray1<f64>, window: usize) -> PyResult<f64> {
        let prices = prices.as_slice()?;
        
        if prices.len() < 2 { return Ok(0.0); }

        let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
        let end = returns.len();
        let start = end.saturating_sub(window);
        let slice = &returns[start..];
        
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let variance = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / slice.len() as f64;

        Ok(variance.sqrt())
    }

    pub fn fast_sharpe_ratio(&self, returns: PyReadonlyArray1<f64>, risk_free_rate: f64) 
        -> PyResult<f64> {
        let returns = returns.as_slice()?;
        
        if returns.len() < 2 { return Ok(0.0); }

        let n = returns.len() as f64;
        let (sum, sum_sq) = returns.iter().fold((0.0, 0.0), |(s, sq), &r| (s + r, sq + r * r));
        let mean = sum / n;
        let std_dev = (sum_sq / n - mean * mean).sqrt();
        
        if std_dev == 0.0 { Ok(0.0) } else { Ok((mean - risk_free_rate) / std_dev) }
    }

    pub fn fast_maximum_drawdown(&self, returns: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let returns = returns.as_slice()?;
        
        if returns.is_empty() { return Ok(0.0); }

        let (_, max_dd) = returns.iter().fold((1.0, 1.0, 0.0), |(cum, peak, max_dd), &ret| {
            let new_cum = cum * (1.0 + ret);
            let new_peak = peak.max(new_cum);
            let dd = (new_peak - new_cum) / new_peak;
            (new_cum, new_peak, max_dd.max(dd))
        });

        Ok(max_dd)
    }

    pub fn fast_kelly_criterion(&self, returns: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let returns = returns.as_slice()?;
        
        if returns.len() < 10 { return Ok(0.0); }

        let (wins, losses, win_count) = returns.iter()
            .fold((0.0, 0.0, 0), |(w, l, wc), &r| {
                if r > 0.0 { (w + r, l, wc + 1) } else { (w, l + r, wc) }
            });

        let loss_count = returns.len() - win_count;
        if win_count == 0 || loss_count == 0 { return Ok(0.0); }

        let win_rate = win_count as f64 / returns.len() as f64;
        let avg_win = wins / win_count as f64;
        let avg_loss = -losses / loss_count as f64;
        let b = avg_win / avg_loss;

        Ok(((b * win_rate - (1.0 - win_rate)) / b).max(0.0).min(0.25))
    }

    pub fn parallel_feature_calculation(&self, prices: PyReadonlyArray2<f64>, 
                                       volumes: PyReadonlyArray2<f64>) 
        -> PyResult<Vec<Vec<f64>>> {
        let prices = prices.as_array();
        let volumes = volumes.as_array();
        
        let (rows, _) = prices.dim();
        
        let results: Vec<Vec<f64>> = (0..rows).into_par_iter().map(|i| {
            let price_row = prices.row(i);
            let volume_row = volumes.row(i);
            
            let mut features = Vec::with_capacity(50);
            let len = price_row.len();
            
            if len >= 20 {
                let (sum5, sum20) = (len-5..len).zip(len-20..)
                    .fold((0.0, 0.0), |(s5, s20), (i5, i20)| {
                        (s5 + price_row[i5], s20 + price_row[i20])
                    });
                let (sma5, sma20) = (sum5 / 5.0, sum20 / 20.0);
                let current = price_row[len - 1];
                features.extend([current / sma5, current / sma20, (sma5 - sma20) / sma20]);
            }
            
            if volume_row.len() >= 10 {
                let vol_avg = volume_row.slice(s![-10..]).mean().unwrap_or(1.0);
                features.push(volume_row[volume_row.len() - 1] / vol_avg);
            }
            
            if len >= 10 {
                let (sum_ret, sum_sq) = price_row.windows(2).fold((0.0, 0.0), |(sr, ssq), w| {
                    let ret = (w[1] / w[0]).ln();
                    (sr + ret, ssq + ret * ret)
                });
                let n = (len - 1) as f64;
                let mean = sum_ret / n;
                features.push((sum_sq / n - mean * mean).sqrt());
            }
            
            if len >= 5 {
                features.push((price_row[len - 1] / price_row[len - 5]) - 1.0);
            }
            
            features.resize(50, 0.0);
            features
        }).collect();

        Ok(results)
    }

    fn calculate_ema_series(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.is_empty() { return vec![]; }

        let alpha = 2.0 / (period as f64 + 1.0);
        let beta = 1.0 - alpha;
        let mut ema = prices[0];
        let mut results = Vec::with_capacity(prices.len());
        results.push(ema);

        for &price in &prices[1..] {
            ema = alpha * price + beta * ema;
            results.push(ema);
        }

        results
    }

    fn calculate_ema_from_series(&self, values: &[f64], period: usize) -> Vec<f64> {
        if values.is_empty() { return vec![]; }

        let alpha = 2.0 / (period as f64 + 1.0);
        let beta = 1.0 - alpha;
        let mut ema = values[0];
        let mut results = Vec::with_capacity(values.len());
        results.push(ema);

        for &value in &values[1..] {
            ema = alpha * value + beta * ema;
            results.push(ema);
        }

        results
    }
}

#[pyfunction]
pub fn fast_portfolio_optimization(returns: PyReadonlyArray2<f64>, 
                                  risk_aversion: f64) 
    -> PyResult<Vec<f64>> {
    let returns_array = returns.as_array();
    let (n_assets, n_periods) = returns_array.dim();
    
    if n_assets == 0 || n_periods < 2 {
        return Ok(vec![1.0 / n_assets as f64; n_assets]);
    }

    let mean_returns: Vec<f64> = (0..n_assets).into_par_iter().map(|i| {
        returns_array.row(i).iter().sum::<f64>() / n_periods as f64
    }).collect();

    let covariance: Vec<Vec<f64>> = (0..n_assets).into_par_iter().map(|i| {
        (0..n_assets).map(|j| {
            let (returns_i, returns_j) = (returns_array.row(i), returns_array.row(j));
            let (mean_i, mean_j) = (mean_returns[i], mean_returns[j]);
            
            returns_i.iter().zip(returns_j.iter())
                .map(|(&ri, &rj)| (ri - mean_i) * (rj - mean_j))
                .sum::<f64>() / (n_periods - 1) as f64
        }).collect()
    }).collect();

    let total_mean: f64 = mean_returns.iter().sum();
    let mut weights: Vec<f64> = if total_mean != 0.0 {
        mean_returns.iter()
            .map(|&mr| (1.0 / n_assets as f64) * (1.0 + mr * risk_aversion / total_mean))
            .collect()
    } else {
        vec![1.0 / n_assets as f64; n_assets]
    };

    let total_weight: f64 = weights.iter().sum();
    if total_weight > 0.0 {
        weights.iter_mut().for_each(|w| *w /= total_weight);
    }

    Ok(weights)
}

#[pymodule]
fn fast_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FastMathEngine>()?;
    m.add_function(wrap_pyfunction!(fast_portfolio_optimization, m)?)?;
    Ok(())
}