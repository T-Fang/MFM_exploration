import torch
import os


def get_device():
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # print(torch.cuda.get_device_name(device))
    return device


def set_torch_default(device=None, dtype=None):
    if device is None:
        device = get_device()
    torch.set_default_device(device)

    if dtype is None:
        dtype = torch.float64

    torch.set_default_dtype(dtype)

    return device, dtype
