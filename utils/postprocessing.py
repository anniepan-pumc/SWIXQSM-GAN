import torch
import numpy as np

def zscore(x, eps=1e-12):
    x = x.type(torch.float32)
    mean = x.mean()
    std = x.std(unbiased=False)
    return (x - mean) / (std + eps)

def scaling_to_minus1_1(x, eps=1e-12):
    x = x.type(torch.float32)
    xmin = x.amin(dim=0, keepdim=True)   # [1, D, H, W]
    xmax = x.amax(dim=0, keepdim=True)   # [1, D, H, W]
    return 2 * (x - xmin) / (xmax - xmin + eps) - 1

def minmax_normalization(x):
    x = x.type(torch.float32)
    return (x - x.min()) / (x.max() - x.min())

def robust_minmax_from_gt(real, pred, p_lo=0.5, p_hi=99.5, eps=1e-8):
    real = real.cpu().numpy()
    pred = pred.cpu().numpy()
    lo = np.percentile(real, p_lo)
    hi = np.percentile(real, p_hi)
    real_n   = np.clip(real, lo, hi)
    pred_n = np.clip(pred, lo, hi)
    real_n   = 2*(real_n - lo)/(hi - lo + eps) - 1
    pred_n = 2*(pred_n - lo)/(hi - lo + eps) - 1
    real_n = torch.from_numpy(real_n)
    pred_n = torch.from_numpy(pred_n)
    return real_n.float(), pred_n.float()