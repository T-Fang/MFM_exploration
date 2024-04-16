import torch
import os
from src.basic.constants import DEFAULT_DTYPE


def get_device():
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
        dtype = DEFAULT_DTYPE

    torch.set_default_dtype(dtype)

    return device, dtype


def get_input_args_for_pool(iterable, *args):
    return [(i, *args) for i in iterable]
