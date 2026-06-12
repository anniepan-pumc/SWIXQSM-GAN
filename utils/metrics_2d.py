import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
import SimpleITK as sitk
import nibabel as nib
from scipy import stats
from scipy.ndimage import laplace, gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.fid import FrechetInceptionDistance

def cal_metrics(real_img, fake_img, mask):
    real_img = np.asarray(real_img)
    fake_img = np.asarray(fake_img)
    mask = None if mask is None else np.asarray(mask, dtype=bool)
    return {
        'psnr2d': calculate_psnr(real_img, fake_img),
        'ssim2d': calculate_ssim(real_img, fake_img),
        'correlation2d': calculate_correlation(real_img, fake_img),
        'fid2d': calculate_fid(real_img, fake_img),
        'mae2d': calculate_mae(real_img, fake_img),
        'rmse2d': calculate_rmse(real_img, fake_img),
        'nmse2d': calculate_nmse(real_img, fake_img, mask),
        'nrmse_p99p1_2d': calculate_nrmse_p99p1(real_img, fake_img, mask),
        'pearson_r_mean_over_slices': corr_metrics_2d_slicewise(real_img, fake_img, mask)['pearson_r_mean_over_slices'],
        'r2_mean_over_slices': corr_metrics_2d_slicewise(real_img, fake_img, mask)['r2_mean_over_slices'],
        'hfen2d': calculate_hfen(real_img, fake_img, mask),
    }

def pearson_r(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x*x).sum()) * np.sqrt((y*y).sum())
    return np.nan if denom == 0 else float((x*y).sum() / denom)

def r2_from_linear_fit(x, y):
    # Fit y = a*x + b, return R^2
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = a*x + b
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = np.nan if ss_tot == 0 else float(1 - ss_res/ss_tot)
    return r2, float(a), float(b)
    
def calculate_psnr( img1, img2, mask=None, data_range=None):
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1: 参考图像
        img2: 生成图像
        data_range: 数据范围，如果为 None 则自动计算
    
    Returns:
        float: 平均 PSNR 值
    """
    psnr_values = []
    for i in range(img1.shape[0]):
        if data_range is None:
            dr = max(img1[i].max() - img1[i].min(), img2[i].max() - img2[i].min())
        else:
            dr = data_range
        
        if dr > 0:
            m = np.ones_like(img1[i], dtype=bool) if mask is None else mask[i].astype(bool)
            x = img1[i][m]
            y = img2[i][m]
            if x.size < 10:
                continue
            psnr = peak_signal_noise_ratio(x, y, data_range=dr)
            psnr_values.append(psnr)
    
    return np.mean(psnr_values) if psnr_values else 0.0

def calculate_ssim( img1, img2, mask=None, data_range=None):
    """
    计算 SSIM (Structural Similarity Index)
    
    Args:
        img1: 参考图像
        img2: 生成图像
        data_range: 数据范围，如果为 None 则自动计算
    
    Returns:
        float: 平均 SSIM 值
    """
    ssim_values = []
    for i in range(img1.shape[0]):
        if data_range is None:
            dr = max(img1[i].max() - img1[i].min(), img2[i].max() - img2[i].min())
        else:
            dr = data_range
        
        if dr > 0:
            m = np.ones_like(img1[i], dtype=bool) if mask is None else mask[i].astype(bool)
            x = img1[i][m]
            y = img2[i][m]
            if x.size < 10:
                continue
            ssim = structural_similarity(x, y, data_range=dr, win_size=min(7, min(x.shape)))
            ssim_values.append(ssim)
    
    return np.mean(ssim_values) if ssim_values else 0.0

def corr_metrics_2d_slicewise(gt_vol, pred_vol, mask=None):
    # gt_vol, pred_vol shape: (D, H, W)
    D = gt_vol.shape[0]
    rs, r2s = [], []
    for k in range(D):
        m = np.ones_like(gt_vol[k], dtype=bool) if mask is None else mask[k].astype(bool)
        x = pred_vol[k][m]
        y = gt_vol[k][m]
        if x.size < 10:
            continue
        rs.append(pearson_r(x, y))
        r2, _, _ = r2_from_linear_fit(x, y)
        r2s.append(r2)
    return {"pearson_r_mean_over_slices": float(np.nanmean(rs)),
            "r2_mean_over_slices": float(np.nanmean(r2s))}

def slicewise_pearson_r_r2(img1, img2, axis=0, bg_threshold=-950):
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("img1 and img2 must be same-shape 3D arrays")

    # ROI mask: keep voxels that are not background in either image
    mask = (a > bg_threshold) | (b > bg_threshold)

    a = np.moveaxis(a, axis, 0)
    b = np.moveaxis(b, axis, 0)
    mask = np.moveaxis(mask, axis, 0)

    S = a.shape[0]
    r = np.full(S, np.nan)
    r2 = np.full(S, np.nan)
    n = np.zeros(S, dtype=int)

    for i in range(S):
        x = a[i].ravel()
        y = b[i].ravel()
        ok = np.isfinite(x) & np.isfinite(y) & mask[i].ravel()

        x = x[ok]; y = y[ok]
        n[i] = x.size
        if x.size < 2:
            continue

        x0 = x - x.mean()
        y0 = y - y.mean()
        denom = np.sqrt((x0 @ x0) * (y0 @ y0))
        if denom == 0:
            continue

        ri = (x0 @ y0) / denom
        ri = float(np.clip(ri, -1.0, 1.0))
        r[i] = ri
        r2[i] = ri * ri

    # Fisher-z mean for r
    valid = np.isfinite(r)
    r_mean = np.nan
    if np.any(valid):
        z = np.arctanh(np.clip(r[valid], -0.999999999, 0.999999999))
        r_mean = float(np.tanh(np.mean(z)))

    return {"r_mean": r_mean, "r2_mean": float(np.nanmean(r2)),
            "r_slices": r, "r2_slices": r2, "n_slices": n}



def calculate_correlation( img1, img2):
    """
    计算相关系数
    
    Args:
        img1: 参考图像
        img2: 生成图像
    
    Returns:
        dict: 包含最大最小值相关系数和切片相关系数的字典
    """
    max_list = []
    fake_max_list = []
    min_list = []
    fake_min_list = []
    slice_rs = []
    slice_ps = []
    
    for i in range(img1.shape[0]):
        max_list.append(np.max(img1[i]))
        min_list.append(np.min(img1[i]))
        fake_max_list.append(np.max(img2[i]))
        fake_min_list.append(np.min(img2[i]))
        
        # 计算切片相关系数
        flat1 = img1[i].flatten()
        flat2 = img2[i].flatten()
        if np.std(flat1) > 0 and np.std(flat2) > 0:
            r, p = stats.pearsonr(flat1, flat2)
            slice_rs.append(r)
            slice_ps.append(p)
    
    # 计算最大最小值相关系数
    max_r, max_p = stats.pearsonr(fake_max_list, max_list) if len(fake_max_list) > 1 else (0, 1)
    min_r, min_p = stats.pearsonr(fake_min_list, min_list) if len(fake_min_list) > 1 else (0, 1)
    
    return {
        'max_correlation': max_r,
        'max_p_value': max_p,
        'min_correlation': min_r,
        'min_p_value': min_p,
        'slice_correlation': np.mean(slice_rs) if slice_rs else 0.0,
        'slice_p_value': np.mean(slice_ps) if slice_ps else 1.0
    }

def calculate_fid( real_tensor, fake_tensor):
    """
    计算 FID (Fréchet Inception Distance)
    
    Args:
        real_tensor: 真实图像张量或数组，形状为 (D, H, W) 或 (D, 1, H, W)
        fake_tensor: 生成图像张量或数组，形状为 (D, H, W) 或 (D, 1, H, W)
    
    Returns:
        float: FID 值
    """
    # 转换为 torch tensor（如果是 numpy 数组）
    if isinstance(real_tensor, np.ndarray):
        real_tensor = torch.from_numpy(real_tensor).float()
    if isinstance(fake_tensor, np.ndarray):
        fake_tensor = torch.from_numpy(fake_tensor).float()
    
    # 确保在 CPU 上
    real_tensor = real_tensor.cpu() if hasattr(real_tensor, 'cpu') else real_tensor
    fake_tensor = fake_tensor.cpu() if hasattr(fake_tensor, 'cpu') else fake_tensor
    
    # 如果形状是 (D, H, W)，添加通道维度变成 (D, 1, H, W)
    if real_tensor.ndim == 3:
        real_tensor = real_tensor.unsqueeze(1)
    if fake_tensor.ndim == 3:
        fake_tensor = fake_tensor.unsqueeze(1)
    
    fid_cal = FrechetInceptionDistance(feature=192)
    
    # 归一化到 0-255 范围
    def normalize_to_uint8(tensor):
        # 对每个切片单独归一化
        normalized = []
        for i in range(tensor.shape[0]):
            slice_t = tensor[i]
            min_val = slice_t.min()
            max_val = slice_t.max()
            if max_val > min_val:
                normalized_slice = (slice_t - min_val) / (max_val - min_val) * 255
            else:
                normalized_slice = slice_t * 0
            normalized.append(normalized_slice)
        normalized_tensor = torch.stack(normalized, dim=0)
        return normalized_tensor.clamp(0, 255).to(torch.uint8)
    
    # 归一化并扩展到 3 通道
    real_normalized = normalize_to_uint8(real_tensor)
    fake_normalized = normalize_to_uint8(fake_tensor)
    
    real_expanded = real_normalized.expand(-1, 3, -1, -1)
    fake_expanded = fake_normalized.expand(-1, 3, -1, -1)
    
    fid_cal.update(real_expanded, real=True)
    fid_cal.update(fake_expanded, real=False)
    
    fid_value = fid_cal.compute().item()
    return fid_value

def calculate_mae( img1, img2):
    """计算平均绝对误差 (MAE)"""
    mae_values = []
    for i in range(img1.shape[0]):
        mae = np.mean(np.abs(img1[i] - img2[i]))
        mae_values.append(mae)
    return np.mean(mae_values)

def calculate_rmse( img1, img2):
    """计算均方根误差 (RMSE)"""
    rmse_values = []
    for i in range(img1.shape[0]):
        rmse = np.sqrt(np.mean((img1[i] - img2[i]) ** 2))
        rmse_values.append(rmse)
    return np.mean(rmse_values)

def calculate_nmse(img1, img2, mask=None):
    """
    计算 NMSE (Normalized Mean Squared Error)。
    NMSE = ||pred - gt||^2 / ||gt||^2，在整个 mask 区域上计算。

    Args:
        img1: 参考图像 (gt)，形状 (D, H, W)
        img2: 生成图像 (pred)，形状 (D, H, W)
        mask: 可选，形状 (D, H, W)，True 表示有效区域

    Returns:
        float: NMSE 值
    """
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    
    if mask is None:
        mask = np.ones_like(img1, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    
    x = img1[mask].ravel()  # gt
    y = img2[mask].ravel()  # pred
    
    if x.size < 10:
        return 0.0
    
    eps = np.finfo(np.float64).eps
    norm_sq_gt = np.sum(x ** 2) + eps
    nmse = np.sum((x - y) ** 2) / norm_sq_gt
    return float(nmse)

def calculate_nrmse_p99p1(img1, img2, mask=None):
    """
    计算 NRMSE (Normalized Root Mean Squared Error)，
    使用 p99-p1 范围归一化。
    NRMSE = RMSE / (p99 - p1)，在整个 mask 区域上计算。

    Args:
        img1: 参考图像 (gt)，形状 (D, H, W)
        img2: 生成图像 (pred)，形状 (D, H, W)
        mask: 可选，形状 (D, H, W)，True 表示有效区域

    Returns:
        float: NRMSE (p99-p1) 值
    """
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    
    if mask is None:
        mask = np.ones_like(img1, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    
    x = img1[mask].ravel()  # gt
    y = img2[mask].ravel()  # pred
    
    if x.size < 10:
        return 0.0
    
    # Calculate p99-p1 range from reference (gt) image
    p1 = np.percentile(x, 1)
    p99 = np.percentile(x, 99)
    data_range = p99 - p1
    
    eps = np.finfo(np.float64).eps
    if data_range < eps:
        return float('nan')
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((x - y) ** 2))
    nrmse = rmse / data_range
    return float(nrmse)

def calculate_hfen(img1, img2, mask=None, sigma=1.5):
    """
    计算 HFEN (High-Frequency Error Norm)。
    使用拉普拉斯高斯滤波器 (LoG) 来衡量高频误差。
    
    返回的是 RMSE 形式的 HFEN：sqrt(mean((LoG(pred) - LoG(gt))^2))
    这种形式更加稳定，不会因为参考图像高频分量小而爆炸。

    Args:
        img1: 参考图像 (gt)，形状 (D, H, W) 或 (H, W)
        img2: 生成图像 (pred)，形状同 img1
        mask: 可选，形状同 img1，True 表示有效区域
        sigma: 高斯滤波器的标准差，默认 1.5

    Returns:
        float: HFEN 值 (越小越好)
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    if img1.ndim == 2:
        # 单张2D图像
        img1 = img1[np.newaxis, ...]
        img2 = img2[np.newaxis, ...]
        if mask is not None:
            mask = mask[np.newaxis, ...]
    
    if mask is None:
        mask = np.ones_like(img1, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    
    hfen_values = []
    for i in range(img1.shape[0]):
        # 先高斯平滑，再拉普拉斯算子 (LoG)
        log_gt = laplace(gaussian_filter(img1[i], sigma=sigma))
        log_pred = laplace(gaussian_filter(img2[i], sigma=sigma))
        
        m = mask[i].astype(bool)
        log_gt_masked = log_gt[m]
        log_pred_masked = log_pred[m]
        
        if log_gt_masked.size < 10:
            continue
        
        # HFEN (RMSE形式) = sqrt(mean((LoG(pred) - LoG(gt))^2))
        hfen = np.sqrt(np.mean((log_pred_masked - log_gt_masked) ** 2))
        hfen_values.append(hfen)
    
    return float(np.mean(hfen_values)) if hfen_values else 0.0