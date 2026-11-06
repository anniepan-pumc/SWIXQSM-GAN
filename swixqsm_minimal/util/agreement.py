"""Agreement of two fixed methods on independent participant ROI scalars.

Each input position must already represent one equally weighted, independent
participant. The caller aggregates repeated visits and records exclusions; this
module never drops pairs, normalizes measurements, or treats voxels as subjects.
"""

from numbers import Integral, Real

import numpy as np


_METRICS = ("bias", "difference_sd", "loa_lower", "loa_upper", "pearson_r", "icc_a1")
_LOA_Z = 1.96
_ICC_MIN_N = 3  # Conservative reporting policy, not the mathematical minimum.
_BOOTSTRAP_MIN_N = 3  # Conservative CI reporting policy, not a validity guarantee.
_SOURCES = [
    {
        "title": "Koo and Li (2016), ICC model, type and definition; Table 3",
        "url": "https://doi.org/10.3348/kjr.2016.17.5.623",
        "doi": "10.1016/j.jcm.2016.02.012",
    },
    {
        "title": "Bland and Altman (1986), Statistical methods for assessing agreement",
        "url": "https://www-users.york.ac.uk/~mb55/meas/ba.htm",
        "doi": "10.1016/S0140-6736(86)90837-8",
    },
    {
        "title": "SciPy bootstrap documentation, paired resampling and percentile intervals",
        "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html",
    },
]


def _vector(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a one-dimensional real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite float64 values; record exclusions before calling")
    return array


def _finite_result(values, statuses, name, value):
    value = float(value)
    if np.isfinite(value):
        values[name], statuses[name] = value, "defined"
    else:
        values[name], statuses[name] = None, "numerical_overflow"


def _common_scaled_pair(reference, prediction):
    """Apply only a common affine change for stable dimensionless statistics."""
    # Common translation preserves absolute agreement, unlike subtracting each
    # method's own mean. Center first to retain small variation on a large offset.
    with np.errstate(over="ignore", invalid="ignore"):
        x = reference-reference[0]
        y = prediction-reference[0]
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        scale = max(float(np.max(np.abs(reference))), float(np.max(np.abs(prediction))))
        x, y = reference/scale, prediction/scale
    scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    if scale:
        x, y = x/scale, y/scale
    return x, y


def _point_estimates(reference, prediction):
    n = reference.size
    values = dict.fromkeys(_METRICS)
    statuses = dict.fromkeys(_METRICS, "insufficient_n")
    statuses["icc_a1"] = "insufficient_n_for_icc_policy"
    if n == 0:
        return values, statuses

    # Rescaling is numerical arithmetic only: dimensional outputs are converted
    # back to the original input units, without clipping or removing the bias.
    with np.errstate(over="ignore", invalid="ignore"):
        difference = prediction-reference
    if np.all(np.isfinite(difference)):
        scale = float(np.max(np.abs(difference))) or 1.0
        difference = difference/scale
    else:
        scale = max(float(np.max(np.abs(reference))), float(np.max(np.abs(prediction))))
        difference = prediction/scale-reference/scale
    mean_difference = float(np.mean(difference))
    with np.errstate(over="ignore", invalid="ignore"):
        _finite_result(values, statuses, "bias", mean_difference*scale)
        if n >= 2:
            difference_sd = float(np.sqrt(np.sum((difference-mean_difference)**2)/(n-1)))
            _finite_result(values, statuses, "difference_sd", difference_sd*scale)
            _finite_result(values, statuses, "loa_lower", (mean_difference-_LOA_Z*difference_sd)*scale)
            _finite_result(values, statuses, "loa_upper", (mean_difference+_LOA_Z*difference_sd)*scale)

    if n < 2:
        return values, statuses
    reference_constant = bool(np.all(reference == reference[0]))
    prediction_constant = bool(np.all(prediction == prediction[0]))
    if reference_constant or prediction_constant:
        statuses["pearson_r"] = ("constant_both" if reference_constant and prediction_constant else
                                 "constant_reference" if reference_constant else "constant_prediction")
    else:
        # Correlation is invariant to separate positive scales and translations.
        # Use each method's own scale here so a large method bias cannot make
        # participant variation numerically vanish in the Pearson calculation.
        x, _ = _common_scaled_pair(reference, reference)
        y, _ = _common_scaled_pair(prediction, prediction)
        x, y = x-np.mean(x), y-np.mean(y)
        norm_x, norm_y = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        if norm_x == 0 or norm_y == 0:
            statuses["pearson_r"] = "numerically_degenerate_variance"
        else:
            r = float(np.dot(x/norm_x, y/norm_y))
            # Only correlation's possible last-bit roundoff is bounded. ICC is
            # deliberately not clipped, including estimates below zero.
            _finite_result(values, statuses, "pearson_r", np.clip(r, -1., 1.))

    if n < _ICC_MIN_N:
        return values, statuses
    if reference_constant and prediction_constant:
        if reference[0] == prediction[0]:
            statuses["icc_a1"] = "degenerate_denominator"
        else:
            # Formula-defined zero: the two methods differ, and neither
            # distinguishes participants. This is not evidence of reliability.
            values["icc_a1"], statuses["icc_a1"] = 0., "defined"
        return values, statuses

    x, y = _common_scaled_pair(reference, prediction)
    measurements = np.column_stack((x, y))
    row_mean = np.mean(measurements, axis=1)
    column_mean = np.mean(measurements, axis=0)
    grand_mean = float(np.mean(measurements))
    k = 2
    ms_rows = float(k*np.sum((row_mean-grand_mean)**2)/(n-1))
    ms_columns = float(n*np.sum((column_mean-grand_mean)**2)/(k-1))
    residual = measurements-row_mean[:, None]-column_mean[None, :]+grand_mean
    ms_error = float(np.sum(residual**2)/((n-1)*(k-1)))
    denominator = ms_rows+(k-1)*ms_error+(k/n)*(ms_columns-ms_error)
    if not np.isfinite(denominator) or denominator <= 0:
        statuses["icc_a1"] = "degenerate_denominator"
    else:
        _finite_result(values, statuses, "icc_a1", (ms_rows-ms_error)/denominator)
    return values, statuses


def paired_agreement(reference, prediction, *, bootstrap_reps=2000, seed=42, confidence=0.95):
    """Return JSON-safe point estimates and paired-participant percentile CIs.

    Inputs are equal-length, finite, one-dimensional real arrays in the same
    physical units and reference convention. Empty arrays are allowed and yield
    explicit unavailable estimates. There is no automatic missing-pair deletion.
    One input pair counts as one independent participant with equal weight.

    Differences are ``prediction-reference``; SD uses ``ddof=1``. Normal-based
    95% limits of agreement (LoA) are ``bias +/- 1.96*SD`` and are not confidence
    intervals for the mean. ``confidence`` controls the bootstrap CI for each of
    the six point estimates, including the two LoA endpoints, without changing
    the LoA's fixed 95% nominal coverage.

    ICC(A,1) is two-way mixed, two fixed methods, absolute agreement, single
    measurement: ``(MSR-MSE)/(MSR+(k-1)*MSE+k/n*(MSC-MSE))``, ``k=2``.
    ICC requires n>=3 here as a conservative reporting policy, not a mathematical
    minimum. Negative estimates are retained. Pearson correlation alone does not
    measure agreement. Constant columns make Pearson undefined; distinct constant
    columns give formula-defined ICC=0, while identical constants have zero ICC
    denominator. Such constant data cannot show participant discrimination.

    Bootstrap draws the same n participant indices for both methods using a
    local NumPy Generator. ``bootstrap_reps=0`` disables resampling. No hard
    replication cap is imposed; the caller sets its computational budget.
    Memory is O(n + 6*bootstrap_reps), without a replicates-by-n sample matrix.
    CIs require n>=3 as a conservative reporting policy. Descriptive bias at n=1
    and SD/LoA at n=2 remain available; repeatedly drawing one participant cannot
    estimate population uncertainty. Undefined resamples are counted, never
    redrawn. If some resamples are
    undefined, intervals use only defined replicates and say ``partial_defined``;
    they are conditional summaries with a stated degeneracy limitation. Fewer
    than two defined replicates produce no interval. An undefined point estimate
    never acquires a CI from resamples, even if some resamples are defined.
    """
    reference, prediction = _vector(reference, "reference"), _vector(prediction, "prediction")
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction must have equal length")
    if isinstance(bootstrap_reps, (bool, np.bool_)) or not isinstance(bootstrap_reps, Integral) or bootstrap_reps < 0:
        raise ValueError("bootstrap_reps must be a nonnegative integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, Real) or
            not np.isfinite(confidence) or not 0 < confidence < 1):
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    bootstrap_reps, seed, confidence = int(bootstrap_reps), int(seed), float(confidence)
    n = int(reference.size)
    estimates, statuses = _point_estimates(reference, prediction)
    samples = {name: [] for name in _METRICS}
    undefined_reasons = {name: {} for name in _METRICS}
    rng = np.random.default_rng(seed)
    if n == 0 and bootstrap_reps:
        # Every empty resample has the same unavailable statistics; count them
        # without spending O(reps) work when callers only validate their options.
        for name in _METRICS:
            undefined_reasons[name][statuses[name]] = bootstrap_reps
    for _ in range(bootstrap_reps if n else 0):
        indices = rng.integers(0, n, size=n)
        values, sample_status = _point_estimates(reference[indices], prediction[indices])
        for name in _METRICS:
            if values[name] is not None:
                samples[name].append(values[name])
            else:
                reason = sample_status[name]
                undefined_reasons[name][reason] = undefined_reasons[name].get(reason, 0)+1

    bootstrap_metrics = {}
    quantiles = ((1-confidence)/2, 1-(1-confidence)/2)
    for name in _METRICS:
        valid = len(samples[name])
        interval = {
            "lower": None, "upper": None, "status": "disabled",
            "valid_replicates": valid, "undefined_replicates": bootstrap_reps-valid,
            "undefined_reason_counts": undefined_reasons[name],
        }
        if bootstrap_reps:
            if estimates[name] is None:
                interval["status"] = "point_estimate_undefined"
            elif n < _BOOTSTRAP_MIN_N:
                interval["status"] = "insufficient_n_for_bootstrap_policy"
            elif valid < 2:
                interval["status"] = "insufficient_defined_replicates"
            else:
                # The linear percentile convention is explicit and reproducible.
                # np.quantile's interpolation can overflow for extreme opposite
                # endpoints; positive rescaling preserves these quantiles.
                sample_values = np.asarray(samples[name], dtype=np.float64)
                scale = float(np.max(np.abs(sample_values))) or 1.
                with np.errstate(over="ignore", invalid="ignore"):
                    bounds = np.quantile(sample_values/scale, quantiles, method="linear")*scale
                if np.all(np.isfinite(bounds)):
                    interval.update(lower=float(bounds[0]), upper=float(bounds[1]),
                                    status="defined" if valid == bootstrap_reps else "partial_defined")
                else:
                    interval["status"] = "numerical_overflow"
        bootstrap_metrics[name] = interval

    return {
        "schema": "swixqsm.paired_agreement", "version": 1,
        "n_participants": n, "estimates": estimates, "metric_status": statuses,
        "bootstrap": {
            "method": "paired_participant_percentile", "requested_replicates": bootstrap_reps,
            "seed": seed, "confidence": confidence, "quantile_method": "linear",
            "minimum_participants_for_ci": _BOOTSTRAP_MIN_N,
            "minimum_n_is_reporting_policy": True,
            "partial_defined_policy": "percentiles_of_defined_replicates_with_counts; no_redraw",
            "metrics": bootstrap_metrics,
        },
        "definitions": {
            "independent_unit": "participant", "weighting": "equal_participant",
            "difference": "prediction-reference", "difference_sd_ddof": 1,
            "dimensional_metrics_units": "same_as_input",
            "loa_nominal_coverage": 0.95, "loa_multiplier": _LOA_Z,
            "loa_is_mean_confidence_interval": False,
            "icc": {
                "name": "ICC(A,1)", "model": "two_way_mixed", "methods": "fixed",
                "definition": "absolute_agreement", "measurement": "single", "k": 2,
                "formula": "(MSR-MSE)/(MSR+(k-1)*MSE+(k/n)*(MSC-MSE))",
                "MSR": "participant_row_mean_square", "MSC": "fixed_method_column_mean_square",
                "MSE": "two_way_additive_residual_mean_square",
                "minimum_n_reporting_policy": _ICC_MIN_N,
                "minimum_n_policy_is_mathematical_minimum": False,
                "negative_estimates_clipped": False,
                "distinct_constant_columns": "formula_defined_zero; no_participant_discrimination",
            },
        },
        "assumptions": [
            "Each pair is one independent participant; repeated visits must be aggregated by the caller.",
            "Both methods use the same physical units, ROI definition and susceptibility reference.",
            "Normal-based LoA assume approximately normal differences and stable bias and SD across measurement magnitude.",
            "ICC interpretation concerns these two fixed methods and the sampled participant population.",
        ],
        "limitations": [
            "Assumptions and participant independence are declared, not proved or automatically tested here.",
            "Pearson association does not establish agreement or interchangeability.",
            "LoA describe differences, whereas bootstrap intervals describe uncertainty of estimated statistics.",
            "Percentile intervals are pointwise; no simultaneous multi-ROI or multi-method coverage is claimed.",
            "Partial-defined intervals condition on defined resamples and can be unstable; counts must accompany them.",
            "Small samples and constant or nearly constant measurements can make agreement estimates unstable or undefined.",
            "No clinical acceptability threshold, hypothesis test, or automatic bias correction is applied.",
        ],
        "sources": [dict(source) for source in _SOURCES],
    }
