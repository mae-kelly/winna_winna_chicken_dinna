# macOS Compatibility Patch for Trading System
import platform
import numpy as cp  # Use numpy instead of cupy
import pandas as cdf  # Use pandas instead of cudf

IS_MACOS = platform.system() == "Darwin"
GPU_AVAILABLE = False

class CPUAcceleration:
    @staticmethod
    def array(data):
        return cp.array(data)
    
    @staticmethod
    def dataframe(data):
        return cdf.DataFrame(data)

def patch_test_system():
    return {}
