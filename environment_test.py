import sys
import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import sklearn

def test_environment():
    """Test all critical components of the environment"""
    
    print("=== CLARITy Environment Test ===\n")
    
    # Python version
    print(f"Python: {sys.version}")
    
    # Core ML libraries
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    
    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    
    # Image processing
    print(f"OpenCV: {cv2.__version__}")
    print(f"Pillow: {Image.__version__}")
    
    # CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Memory optimization test
    if torch.cuda.is_available():
        print("\nTesting mixed precision training...")
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        print("✓ Mixed precision support available")
    
    print("\n=== Environment Setup Complete! ===")

if __name__ == "__main__":
    test_environment()
