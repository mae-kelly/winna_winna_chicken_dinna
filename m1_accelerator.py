"""
M1 GPU Accelerator for Trading System - WORKING VERSION
"""
import torch
import numpy as np
import platform

class M1Accelerator:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.gpu_available = torch.backends.mps.is_available()
        
        if self.gpu_available:
            print(f"🚀 M1 GPU accelerator initialized: {self.device}")
        else:
            print(f"⚠️  CPU fallback initialized: {self.device}")
    
    def array(self, data):
        """Create tensor on M1 GPU"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def numpy(self, tensor):
        """Convert back to numpy"""
        if tensor.device.type == 'mps':
            return tensor.cpu().numpy()
        return tensor.numpy()
    
    def moving_average(self, data, window):
        """GPU-accelerated moving average - FIXED VERSION"""
        if isinstance(data, np.ndarray):
            data = self.array(data)
        
        # Use unfold instead of padding for 1D tensors
        if len(data.shape) == 1:
            # Ensure we have enough data points
            if len(data) < window:
                return data.mean().expand(len(data))
            
            # Use unfold to create sliding windows
            unfolded = data.unfold(0, window, 1)
            result = unfolded.mean(dim=1)
            
            # Pad the beginning to match original length
            padding = torch.full((window-1,), result[0], device=self.device)
            result = torch.cat([padding, result])
        else:
            result = data
        
        if self.gpu_available:
            torch.mps.synchronize()
        
        return result
    
    def rsi(self, prices, period=14):
        """GPU-accelerated RSI calculation"""
        if isinstance(prices, np.ndarray):
            prices = self.array(prices)
        
        # Calculate price changes
        delta = prices[1:] - prices[:-1]
        
        # Separate gains and losses
        gains = torch.where(delta > 0, delta, torch.tensor(0.0, device=self.device))
        losses = torch.where(delta < 0, -delta, torch.tensor(0.0, device=self.device))
        
        # Calculate simple moving averages for gains and losses
        if len(gains) >= period:
            avg_gains = self.moving_average(gains, period)
            avg_losses = self.moving_average(losses, period)
            
            # Calculate RS and RSI
            rs = avg_gains / (avg_losses + 1e-8)
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = torch.full_like(prices[1:], 50.0)  # Default RSI value
        
        if self.gpu_available:
            torch.mps.synchronize()
        
        return rsi

# Global accelerator instance
m1_gpu = M1Accelerator()

# Compatibility aliases to replace cupy/cudf
cp = m1_gpu  # Replace cupy
cdf = None   # cudf not needed with our approach

GPU_AVAILABLE = m1_gpu.gpu_available
IS_MACOS = True
