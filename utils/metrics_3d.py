import numpy as np
from typing import Optional, Callable, Union
from scipy.ndimage import gaussian_filter, laplace

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

def cal_metrics(gt_vol, pred_vol, mask=None):
  gt_vol = np.asarray(gt_vol)
  pred_vol = np.asarray(pred_vol)
  if gt_vol.shape != pred_vol.shape:
      raise ValueError(f"图像形状不匹配: gt {gt_vol.shape} vs pred {pred_vol.shape}")

  if mask is None:
      mask = np.ones_like(gt_vol, dtype=bool)
  else:
      mask = np.asarray(mask, dtype=bool)
      if mask.shape != gt_vol.shape:
          raise ValueError(f"mask 形状与图像不匹配: mask {mask.shape} vs gt {gt_vol.shape}")
  x = pred_vol[mask]  # choose x = pred
  y = gt_vol[mask]    # choose y = gt
  r = pearson_r(x, y)
  r2, a, b = r2_from_linear_fit(x, y)
  out = {"pearson_r": r, "r2": r2, "slope": a, "intercept": b}
  out["nmse_3d"] = nmse_3d(gt_vol, pred_vol, mask)
  out["nrmse_p99p1_3d"] = nrmse_p99p1_3d(gt_vol, pred_vol, mask)
  return out

def mae_3d(a: np.ndarray, b: np.ndarray) -> float:
  """Returns the Mean Absolute Error between `a` and `b`.

  Args:
    a: First image (or set of images), shape (..., D, H, W) or (..., D, H, W, C).
    b: Second image (or set of images), same shape as a.

  Returns:
    MAE between `a` and `b`.
  """
  a = np.asarray(a)
  b = np.asarray(b)
  
  if a.shape != b.shape:
    raise ValueError(f"图像形状不匹配: a {a.shape} vs b {b.shape}")
  
  if a.ndim < 3:
    raise ValueError(f"期望至少3D数组，但得到 {a.ndim}D 数组")
  
  # 支持 3D, 4D, 5D 数组
  if a.ndim == 3:
    return float(np.mean(np.abs(a - b)))
  elif a.ndim == 4:
    return float(np.mean(np.abs(a - b), axis=(-3, -2, -1)))
  else:  # 5D
    return float(np.mean(np.abs(a - b), axis=(-4, -3, -2, -1)))

def mse_3d(a: np.ndarray, b: np.ndarray) -> float:
  """Returns the Mean Squared Error between `a` and `b`.

  Args:
    a: First image (or set of images), shape (..., D, H, W) or (..., D, H, W, C).
    b: Second image (or set of images), same shape as a.

  Returns:
    MSE between `a` and `b`.
  """
  a = np.asarray(a)
  b = np.asarray(b)
  
  if a.shape != b.shape:
    raise ValueError(f"图像形状不匹配: a {a.shape} vs b {b.shape}")
  
  if a.ndim < 3:
    raise ValueError(f"期望至少3D数组，但得到 {a.ndim}D 数组")
  
  if a.ndim == 3:
    return float(np.mean((a - b) ** 2))
  elif a.ndim == 4:
    return float(np.mean((a - b) ** 2, axis=(-3, -2, -1)))
  else:  # 5D
    return float(np.mean((a - b) ** 2, axis=(-4, -3, -2, -1)))

def nmse_3d(
  a: np.ndarray,
  b: np.ndarray,
  mask: Optional[np.ndarray] = None,
) -> float:
  """Returns the Normalized Mean Squared Error: ||a - b||^2 / ||a||^2.

  Args:
    a: Reference image (gt), shape (..., D, H, W) or (..., D, H, W, C).
    b: Predicted image, same shape as a.
    mask: Optional boolean mask, same shape as a. If None, use all voxels.

  Returns:
    NMSE between a and b.
  """
  a = np.asarray(a)
  b = np.asarray(b)
  if a.shape != b.shape:
    raise ValueError(f"图像形状不匹配: a {a.shape} vs b {b.shape}")
  if a.ndim < 3:
    raise ValueError(f"期望至少3D数组，但得到 {a.ndim}D 数组")

  if mask is not None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != a.shape:
      raise ValueError(f"mask 形状与图像不匹配: mask {mask.shape} vs a {a.shape}")
    a = a[mask]
    b = b[mask]

  eps = np.finfo(np.float64).eps
  norm_sq_ref = np.sum(a ** 2) + eps
  nmse = np.sum((a - b) ** 2) / norm_sq_ref
  return float(nmse)

def nrmse_p99p1_3d(
  a: np.ndarray,
  b: np.ndarray,
  mask: Optional[np.ndarray] = None,
) -> float:
  """Returns the Normalized Root Mean Squared Error using p99-p1 range.

  NRMSE = RMSE / (p99 - p1), where p99 and p1 are the 99th and 1st percentiles
  of the reference image (gt).

  Args:
    a: Reference image (gt), shape (..., D, H, W) or (..., D, H, W, C).
    b: Predicted image, same shape as a.
    mask: Optional boolean mask, same shape as a. If None, use all voxels.

  Returns:
    NRMSE (p99-p1 normalized) between a and b.
  """
  a = np.asarray(a)
  b = np.asarray(b)
  if a.shape != b.shape:
    raise ValueError(f"图像形状不匹配: a {a.shape} vs b {b.shape}")
  if a.ndim < 3:
    raise ValueError(f"期望至少3D数组，但得到 {a.ndim}D 数组")

  if mask is not None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != a.shape:
      raise ValueError(f"mask 形状与图像不匹配: mask {mask.shape} vs a {a.shape}")
    a_masked = a[mask]
    b_masked = b[mask]
  else:
    a_masked = a.ravel()
    b_masked = b.ravel()

  # Calculate p99-p1 range from reference image
  p1 = np.percentile(a_masked, 1)
  p99 = np.percentile(a_masked, 99)
  data_range = p99 - p1

  eps = np.finfo(np.float64).eps
  if data_range < eps:
    return float('nan')

  # Calculate RMSE
  rmse = np.sqrt(np.mean((a_masked - b_masked) ** 2))
  nrmse = rmse / data_range
  return float(nrmse)

def psnr_3d(a: np.ndarray,
            b: np.ndarray,
            data_range: Union[tuple[float, float], float] = 1.0) -> float:
  """Returns the Peak Signal-to-Noise Ratio between `a` and `b`.

  Assumes that the dynamic range of the images (the difference between the
  maximum and the minimum allowed values) is 1.0.

  Args:
    a: First image (or set of images).
    b: Second image (or set of images).
    data_range: Range of the data of a and b.

  Returns:
    PSNR in decibels between `a` and `b`.
  """
  a = np.asarray(a)
  b = np.asarray(b)
  
  if a.shape != b.shape:
    raise ValueError(f"图像形状不匹配: a {a.shape} vs b {b.shape}")
  
  if isinstance(data_range, tuple):
    data_range = data_range[1] - data_range[0]

  mse_val = mse_3d(a, b)
  if mse_val == 0:
    return float('inf')
  
  return float(20 * np.log10(data_range) - 10 * np.log10(mse_val))

def ssim_3d(
    a: np.ndarray,
    b: np.ndarray,
    *,
    data_range: Union[float, tuple[float, float]] = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_map: bool = False,
    filter_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Union[float, np.ndarray]:
  """Computes the structural similarity index (SSIM) between image pairs.

  This function is based on the standard SSIM implementation from:
  Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
  "Image quality assessment: from error visibility to structural similarity",
  in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.

  Note: the true SSIM is only defined on grayscale. This function does not
  perform any colorspace transform. If the input is in a color space, then it
  will compute the average SSIM.

  Args:
    a: First image (or set of images), shape (..., D, H, W) or (..., D, H, W, C).
    b: Second image (or set of images), same shape as a.
    data_range: Range of the data.
    filter_size: Window size (>= 1). Image dims must be at least this small.
    filter_sigma: The bandwidth of the Gaussian used for filtering (> 0.).
    k1: One of the SSIM dampening parameters (> 0.).
    k2: One of the SSIM dampening parameters (> 0.).
    return_map: If True, will cause the per-pixel SSIM "map" to be returned.
    filter_fn: An optional argument for overriding the filter function used by
      SSIM, which would otherwise be a 3D Gaussian blur specified by filter_sigma.

  Returns:
    Each image's mean SSIM, or a tensor of individual values if `return_map`.
  """
  a = np.asarray(a)
  b = np.asarray(b)
  
  if a.shape != b.shape:
    raise ValueError(f"图像形状不匹配: a {a.shape} vs b {b.shape}")
  
  if a.ndim < 3:
    raise ValueError(f"期望至少3D数组，但得到 {a.ndim}D 数组")

  if isinstance(data_range, tuple):
    data_range = data_range[1] - data_range[0]

  # 定义默认的3D高斯滤波函数
  if filter_fn is None:
    def filter_fn(z):
      """3D高斯滤波"""
      return gaussian_filter(z, sigma=filter_sigma)

  # 应用滤波
  mu0 = filter_fn(a)
  mu1 = filter_fn(b)
  mu00 = mu0 * mu0
  mu11 = mu1 * mu1
  mu01 = mu0 * mu1
  sigma00 = filter_fn(a**2) - mu00
  sigma11 = filter_fn(b**2) - mu11
  sigma01 = filter_fn(a * b) - mu01

  # Clip the variances and covariances to valid values.
  # Variance must be non-negative:
  epsilon = np.finfo(np.float32).eps ** 2
  sigma00 = np.maximum(epsilon, sigma00)
  sigma11 = np.maximum(epsilon, sigma11)
  sigma01 = np.sign(sigma01) * np.minimum(
      np.sqrt(sigma00 * sigma11), np.abs(sigma01)
  )

  c1 = (k1 * data_range) ** 2
  c2 = (k2 * data_range) ** 2
  numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
  denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
  ssim_map = numer / denom
  
  # 计算平均值，根据数组维度决定平均的轴
  if a.ndim == 3:
    ssim_value = np.mean(ssim_map)
  elif a.ndim == 4:
    ssim_value = np.mean(ssim_map, axis=(-3, -2, -1))
  else:  # 5D
    ssim_value = np.mean(ssim_map, axis=(-4, -3, -2, -1))
  
  return ssim_map if return_map else ssim_value

def hfen_3d(
    a: np.ndarray,
    b: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma: float = 1.5,
) -> float:
  """计算 HFEN (High-Frequency Error Norm)。
  
  使用拉普拉斯高斯滤波器 (LoG) 来衡量高频误差。
  常用于 MRI 图像重建质量评估。
  
  返回的是 RMSE 形式的 HFEN：sqrt(mean((LoG(pred) - LoG(gt))^2))
  这种形式更加稳定，不会因为参考图像高频分量小而爆炸。

  Args:
    a: 参考图像 (gt)，形状 (..., D, H, W) 或 (..., D, H, W, C)。
    b: 预测图像，形状同 a。
    mask: 可选，布尔类型 mask，形状同 a。若为 None，使用所有体素。
    sigma: 高斯滤波器的标准差，默认 1.5。

  Returns:
    HFEN 值 (越小越好)。
  """
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  
  if a.shape != b.shape:
    raise ValueError(f"图像形状不匹配: a {a.shape} vs b {b.shape}")
  if a.ndim < 3:
    raise ValueError(f"期望至少3D数组，但得到 {a.ndim}D 数组")

  # 应用 LoG (Laplacian of Gaussian) 滤波器
  log_a = laplace(gaussian_filter(a, sigma=sigma))
  log_b = laplace(gaussian_filter(b, sigma=sigma))

  if mask is not None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != a.shape:
      raise ValueError(f"mask 形状与图像不匹配: mask {mask.shape} vs a {a.shape}")
    log_a = log_a[mask]
    log_b = log_b[mask]
  else:
    log_a = log_a.ravel()
    log_b = log_b.ravel()

  # HFEN (RMSE形式) = sqrt(mean((LoG(pred) - LoG(gt))^2))
  hfen = np.sqrt(np.mean((log_b - log_a) ** 2))
  return float(hfen)

if __name__ == "__main__":
  x = np.zeros((32, 32, 32))
  y = np.ones((32, 32, 32))
  print(f"SSIM (3D): {ssim_3d(x, y)}")
  print(f"PSNR (3D): {psnr_3d(x, y)}")
  print(f"MAE (3D): {mae_3d(x, y)}")
  print(f"MSE (3D): {mse_3d(x, y)}")
  print(f"NMSE (3D): {nmse_3d(x, y)}")
  print(f"NRMSE_P99P1 (3D): {nrmse_p99p1_3d(x, y)}")
  print(f"HFEN (3D): {hfen_3d(x, y)}")

  x = np.zeros((1, 32, 32, 32))
  y = np.ones((1, 32, 32, 32))
  print(f"\nSSIM (4D): {ssim_3d(x, y)}")
  print(f"PSNR (4D): {psnr_3d(x, y)}")
  print(f"MAE (4D): {mae_3d(x, y)}")
  print(f"MSE (4D): {mse_3d(x, y)}")
  print(f"NMSE (4D): {nmse_3d(x, y)}")
  print(f"NRMSE_P99P1 (4D): {nrmse_p99p1_3d(x, y)}")
  print(f"HFEN (4D): {hfen_3d(x, y)}")
