import torch

def get_device_info():
    """Get GPU device information and availability"""
    cuda_available = torch.cuda.is_available()
    return {
        'available': cuda_available,
        'type': 'CUDA (NVIDIA)' if cuda_available else 'CPU',
        'device': torch.device("cuda" if cuda_available else "cpu")
    } 