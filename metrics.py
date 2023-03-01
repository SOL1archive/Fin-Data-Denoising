import math

import numpy as np
import torch
import torch.functional as F
from sklearn.metrics import mean_squared_error


def snr(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    signal_power = np.mean(y_true**2)

    return 10 * np.log10(signal_power / mse)

def psnr(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    max_channel = np.max(np.concatenate([y_true, y_pred], axis=0))

    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_channel / np.sqrt(mse))

# The code below is from https://gaussian37.github.io/vision-concept-ssim/

def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    # _1d_window : (window_size, 1)
    # sum of _1d_window = 1
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    
    # Converting to 2D  : _1d_window (window_size, 1) @ _1d_window.T (1, window_size)
    # _2d_window : (window_size, window_size)
    # sum of _2d_window = 1
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
     
    # expand _2d_window to window size
    # window : (channel, 1, window_size, window_size)
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window
