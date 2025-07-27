use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

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
}

#[pymodule]
fn fast_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FastMathEngine>()?;
    Ok(())
}
