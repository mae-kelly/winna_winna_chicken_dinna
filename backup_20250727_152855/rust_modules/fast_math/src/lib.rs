use pyo3::prelude::*;
use numpy::PyReadonlyArray1;

#[pyclass]
pub struct FastMathEngine;

#[pymethods]
impl FastMathEngine {
    #[new]
    fn new() -> Self {
        FastMathEngine
    }
    
    fn fast_rsi(&self, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<f64> {
        let prices = prices.as_slice()?;
        if prices.len() < 2 {
            return Ok(50.0);
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        let mut count = 0;
        
        for i in 1..prices.len().min(period + 1) {
            let diff = prices[i] - prices[i-1];
            if diff > 0.0 {
                gains += diff;
            } else {
                losses -= diff;
            }
            count += 1;
        }
        
        if count == 0 || losses == 0.0 {
            return Ok(50.0);
        }
        
        let avg_gain = gains / count as f64;
        let avg_loss = losses / count as f64;
        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        
        Ok(rsi)
    }

    fn fast_bollinger_bands(&self, prices: PyReadonlyArray1<f64>, period: usize, std_dev: f64) 
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

    fn fast_ema(&self, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<f64> {
        let prices = prices.as_slice()?;
        if prices.is_empty() { return Ok(0.0); }

        let alpha = 2.0 / (period as f64 + 1.0);
        let beta = 1.0 - alpha;
        
        Ok(prices.iter().skip(1).fold(prices[0], |ema, &price| alpha * price + beta * ema))
    }

    fn fast_volatility(&self, prices: PyReadonlyArray1<f64>, window: usize) -> PyResult<f64> {
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
}

#[pymodule]
fn fast_math(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastMathEngine>()?;
    Ok(())
}
