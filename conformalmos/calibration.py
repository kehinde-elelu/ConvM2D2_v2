"""Split-conformal absolute-residual quantiles; historical mode is explicit."""
import math
import numpy as np

def quantile(residuals, alpha=0.1, method="finite_sample"):
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Residuals must be a nonempty vector of finite nonnegative values")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if method == "historical":
        return float(np.quantile(values, 1 - alpha)), None
    if method != "finite_sample":
        raise ValueError("Unknown calibration method")
    return (float(np.sort(values)[rank - 1]) if rank <= len(values) else math.inf), rank

def intervals(predictions, radius):
    if math.isnan(radius) or radius < 0:
        raise ValueError("Radius must be nonnegative")
    p = np.asarray(predictions, dtype=float)
    return np.clip(p - radius, 1, 5), np.clip(p + radius, 1, 5)
