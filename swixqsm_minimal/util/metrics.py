"""Masked metrics for 3D maps already expressed in a common physical unit.

No intensity normalization, clipping, referencing, registration, or unit conversion
is performed here. Spatial filters use the original neighbourhood, and only
centres whose complete finite filter support lies inside ``mask`` are evaluated.
"""

import math

import numpy as np
from scipy import ndimage


def _finite_number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _full_support(mask, size):
    # A rectangular minimum filter is a separable, bounded-memory erosion.
    if any(width > extent for width, extent in zip(size, mask.shape)):
        return np.zeros_like(mask, dtype=bool)
    return ndimage.minimum_filter(mask, size=size, mode="constant", cval=0)


def _physical_log(image, sigma_voxels, spacing):
    result = np.zeros_like(image, dtype=np.float64)
    dc_response = 0.
    for axis in range(3):
        order = [0, 0, 0]
        order[axis] = 2
        result += ndimage.gaussian_filter(
            image, sigma=sigma_voxels, order=order,
            mode="constant", cval=0., truncate=4.,
        ) / spacing[axis] ** 2
        # A sampled/truncated Gaussian derivative need not sum to zero,
        # especially when sigma_voxels < 1 (e.g. thick axial slices). Remove
        # its DC response using the matching zero-order Gaussian kernel.
        sigma = sigma_voxels[axis]
        radius = int(4. * sigma + .5)
        positions = np.arange(-radius, radius+1, dtype=np.float64)
        gaussian = np.exp(-.5 * (positions / sigma) ** 2)
        gaussian /= gaussian.sum()
        derivative_sum = np.sum((((positions / sigma) ** 2 - 1.) / sigma ** 2) * gaussian)
        dc_response += derivative_sum / spacing[axis] ** 2
    result -= dc_response * ndimage.gaussian_filter(
        image, sigma=sigma_voxels, mode="constant", cval=0., truncate=4.,
    )
    return result


def physical_metrics(reference, prediction, mask, *, data_range=None,
                     spacing=(1, 1, 1), ssim_window=7, hfen_sigma_mm=1.5):
    """Return strict-JSON-safe physical error and spatial similarity metrics.

    ``mask > 0`` defines the evaluation domain. The three arrays must have the
    same nonempty 3D shape; reference and prediction must be finite inside the
    mask. Values outside it are ignored, including nonfinite values.

    PSNR and SSIM require an explicitly supplied positive, finite ``data_range``
    in the same susceptibility unit. Missing/invalid ranges yield null metrics
    with explanatory statuses. PSNR for identical maps is null with status
    ``infinite``. No data-dependent peak or range is inferred.

    SSIM uses uniform square windows, sample covariance (N/(N-1)), K1=0.01,
    K2=0.03, independently along axis-2 slices. HFEN uses a 3D Gaussian LoG
    (four-sigma truncation and explicit zero-DC kernel correction), with sigma
    in millimetres and derivative scaling by voxel spacing squared. HFEN RMSE
    has units susceptibility/mm^2;
    relative HFEN is ||LoG(pred)-LoG(ref)||_2 / ||LoG(ref)||_2.
    ``spacing`` assumes an orthogonal voxel grid; affine geometry and spatial
    units must be checked by the caller before reducing them to voxel lengths.

    Standard NMSE uses the *uncentred* reference energy. Zero denominators
    produce null and explicit statuses, never epsilon-regularized values.
    Additive ``sufficient_statistics`` support exact voxel/centre-weighted
    aggregation of multiple volumes. Nonrepresentable float64 sums are null
    rather than JSON NaN/Infinity; callers must not silently aggregate them.
    """
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    mask = np.asarray(mask)
    if (reference.ndim != 3 or reference.shape != prediction.shape
            or reference.shape != mask.shape or any(n == 0 for n in reference.shape)):
        raise ValueError("reference, prediction, and mask must share a nonempty 3D shape")
    if (not np.issubdtype(reference.dtype, np.number)
            or not np.issubdtype(prediction.dtype, np.number)
            or np.iscomplexobj(reference) or np.iscomplexobj(prediction)):
        raise ValueError("reference and prediction must be real numeric arrays")
    if not (np.issubdtype(mask.dtype, np.number) or mask.dtype == np.bool_):
        raise ValueError("mask must be numeric or boolean")
    if np.iscomplexobj(mask) or not np.isfinite(mask).all() or np.any(mask < 0):
        raise ValueError("mask must be finite and nonnegative")
    mask = mask > 0
    n_voxels = int(mask.sum())
    if not n_voxels:
        raise ValueError("mask must contain at least one evaluation voxel")
    reference = reference.astype(np.float64, copy=False)
    prediction = prediction.astype(np.float64, copy=False)
    if not np.isfinite(reference[mask]).all() or not np.isfinite(prediction[mask]).all():
        raise ValueError("reference and prediction must be finite inside mask")
    try:
        spacing = tuple(_finite_number(v, "spacing") for v in spacing)
    except TypeError as exc:
        raise ValueError("spacing must contain three positive finite values") from exc
    if len(spacing) != 3:
        raise ValueError("spacing must contain three positive finite values")
    if isinstance(ssim_window, (bool, np.bool_)) or not isinstance(ssim_window, (int, np.integer)):
        raise ValueError("ssim_window must be an odd integer >= 3")
    ssim_window = int(ssim_window)
    if ssim_window < 3 or ssim_window % 2 == 0:
        raise ValueError("ssim_window must be an odd integer >= 3")
    sigma_mm = _finite_number(hfen_sigma_mm, "hfen_sigma_mm")
    sigma_voxels = tuple(sigma_mm / value for value in spacing)
    if any(not math.isfinite(4. * value) for value in sigma_voxels):
        raise ValueError("hfen sigma divided by spacing must be finite")
    radii = tuple(int(4. * value + .5) for value in sigma_voxels)
    statuses = {}
    result = {
        "mae": None, "rmse": None, "bias": None, "mse": None,
        "nmse": None, "psnr_db": None, "ssim": None,
        "hfen_rmse": None, "hfen_relative": None,
        "n_voxels": n_voxels,
        "ref_mean": None, "pred_mean": None, "ref_std": None, "pred_std": None,
        "ssim_centers": 0, "hfen_centers": 0,
        "metric_status": statuses,
        "parameters": {
            "data_range": None, "spacing_mm": list(spacing),
            "ssim_window": ssim_window, "ssim_axis": 2,
            "ssim_covariance": "sample", "hfen_sigma_mm": sigma_mm,
            "hfen_truncate": 4., "hfen_radius_voxels": list(radii),
            "hfen_zero_dc_correction": True,
            "hfen_operator": "zero_dc_corrected_gaussian_second_derivatives",
        },
        "metric_units": {
            "mae": "input_susceptibility", "rmse": "input_susceptibility",
            "bias": "input_susceptibility", "mse": "input_susceptibility^2",
            "hfen_rmse": "input_susceptibility/mm^2", "hfen_relative": "dimensionless",
        },
        "sufficient_statistics": {
            "n": n_voxels, "abs_error_sum": 0., "squared_error_sum": 0.,
            "signed_error_sum": 0., "reference_sum": 0., "reference_energy": 0.,
            "prediction_sum": 0., "prediction_energy": 0.,
            "ssim_sum": 0., "ssim_count": 0,
            "hfen_error_energy": 0., "hfen_reference_energy": 0., "hfen_count": 0,
        },
    }
    statistics = result["sufficient_statistics"]

    def assign(name, value):
        value = float(value)
        if math.isfinite(value):
            result[name] = value
            statuses[name] = "ok"
        else:
            statuses[name] = "numerical_overflow_or_invalid"

    def stat(name, value):
        value = float(value)
        statistics[name] = value if math.isfinite(value) else None

    ref_values, pred_values = reference[mask], prediction[mask]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        error = pred_values - ref_values
        mse = np.mean(error ** 2, dtype=np.float64)
        ref_energy = np.sum(ref_values ** 2, dtype=np.float64)
        for name, value in (
            ("abs_error_sum", np.sum(np.abs(error), dtype=np.float64)),
            ("squared_error_sum", np.sum(error ** 2, dtype=np.float64)),
            ("signed_error_sum", np.sum(error, dtype=np.float64)),
            ("reference_sum", np.sum(ref_values, dtype=np.float64)),
            ("reference_energy", ref_energy),
            ("prediction_sum", np.sum(pred_values, dtype=np.float64)),
            ("prediction_energy", np.sum(pred_values ** 2, dtype=np.float64)),
        ):
            stat(name, value)
        for name, value in (
            ("mae", np.mean(np.abs(error), dtype=np.float64)),
            ("rmse", np.sqrt(mse)), ("bias", np.mean(error, dtype=np.float64)),
            ("mse", mse), ("ref_mean", np.mean(ref_values, dtype=np.float64)),
            ("pred_mean", np.mean(pred_values, dtype=np.float64)),
            ("ref_std", np.std(ref_values, dtype=np.float64)),
            ("pred_std", np.std(pred_values, dtype=np.float64)),
        ):
            assign(name, value)
        if ref_energy == 0:
            statuses["nmse"] = "zero_reference_energy"
        elif not np.isfinite(ref_energy):
            statuses["nmse"] = "numerical_overflow_or_invalid"
        else:
            assign("nmse", np.sum(error ** 2, dtype=np.float64) / ref_energy)

    range_status = "missing_data_range" if data_range is None else "invalid_data_range"
    try:
        peak = _finite_number(data_range, "data_range")
    except ValueError:
        peak = None
    if peak is None:
        statuses["psnr_db"] = range_status
        statuses["ssim"] = range_status
    else:
        result["parameters"]["data_range"] = peak
        if mse == 0:
            statuses["psnr_db"] = "infinite"
        elif np.isfinite(mse):
            assign("psnr_db", 20. * math.log10(peak) - 10. * math.log10(float(mse)))
        else:
            statuses["psnr_db"] = "numerical_overflow_or_invalid"

    # Replacing excluded values only protects computation from NaN/Inf. It has
    # no effect at evaluated centres because their entire support is inside mask.
    ref_image = np.where(mask, reference, 0.)
    pred_image = np.where(mask, prediction, 0.)
    ssim_support = _full_support(mask, (ssim_window, ssim_window, 1))
    result["ssim_centers"] = int(ssim_support.sum())
    if peak is not None:
        if not result["ssim_centers"]:
            statuses["ssim"] = "insufficient_support"
        else:
            size = (ssim_window, ssim_window, 1)
            def mean_filter(image):
                return ndimage.uniform_filter(image, size=size, mode="constant", cval=0.)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                ref_mu, pred_mu = mean_filter(ref_image), mean_filter(pred_image)
                covariance_scale = ssim_window ** 2 / (ssim_window ** 2 - 1)
                ref_var = np.maximum(0., (mean_filter(ref_image ** 2) - ref_mu ** 2) * covariance_scale)
                pred_var = np.maximum(0., (mean_filter(pred_image ** 2) - pred_mu ** 2) * covariance_scale)
                covariance = (mean_filter(ref_image * pred_image) - ref_mu * pred_mu) * covariance_scale
                c1, c2 = (np.float64(.01) * peak) ** 2, (np.float64(.03) * peak) ** 2
                ssim_map = ((2. * ref_mu * pred_mu + c1) * (2. * covariance + c2)
                            / ((ref_mu ** 2 + pred_mu ** 2 + c1) * (ref_var + pred_var + c2)))
                assign("ssim", np.mean(ssim_map[ssim_support], dtype=np.float64))
                stat("ssim_sum", np.sum(ssim_map[ssim_support], dtype=np.float64))
                statistics["ssim_count"] = result["ssim_centers"]

    hfen_support = _full_support(mask, tuple(2 * radius + 1 for radius in radii))
    result["hfen_centers"] = int(hfen_support.sum())
    if not result["hfen_centers"]:
        statuses["hfen_rmse"] = "insufficient_support"
        statuses["hfen_relative"] = "insufficient_support"
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            # A constant is analytically in the corrected LoG null space.
            # Preserve this exact zero without an epsilon energy denominator.
            ref_log = (np.zeros(result["hfen_centers"], dtype=np.float64)
                       if np.ptp(ref_values) == 0 else
                       _physical_log(ref_image, sigma_voxels, spacing)[hfen_support])
            # Linearity permits filtering the error directly, avoiding loss of
            # precision from subtracting two filtered maps with a fixed bias.
            error_image = np.where(mask, pred_image-ref_image, 0.)
            error_log = (np.zeros(result["hfen_centers"], dtype=np.float64)
                         if np.ptp(error) == 0 else
                         _physical_log(error_image, sigma_voxels, spacing)[hfen_support])
            log_error_energy = np.sum(error_log ** 2, dtype=np.float64)
            ref_log_energy = np.sum(ref_log ** 2, dtype=np.float64)
            stat("hfen_error_energy", log_error_energy)
            stat("hfen_reference_energy", ref_log_energy)
            statistics["hfen_count"] = result["hfen_centers"]
            assign("hfen_rmse", np.sqrt(log_error_energy / result["hfen_centers"]))
            if ref_log_energy == 0:
                statuses["hfen_relative"] = "zero_reference_high_frequency_energy"
            elif not np.isfinite(ref_log_energy):
                statuses["hfen_relative"] = "numerical_overflow_or_invalid"
            else:
                assign("hfen_relative", np.sqrt(log_error_energy / ref_log_energy))
    return result
