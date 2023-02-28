import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

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

def ssim(y_true, y_pred, window_size=20):
    scores = []
    for i in range(len(y_true) - window_size):
        score = ssim(y_true[i:i + window_size], y_pred[i:i + window_size])
        scores.append(score)
    return np.mean(scores)
