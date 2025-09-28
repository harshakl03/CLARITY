import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Test tensor creation on GPU
if torch.cuda.is_available():
    x = torch.rand(5, 3).cuda()
    print(f"GPU tensor created successfully: {x.device}")
else:
    print("GPU not available, using CPU")
