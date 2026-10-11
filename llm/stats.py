"""Wilson score interval for a binomial proportion."""
import math


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes out of n trials.

    Returns (low, high), both in [0, 1]. z=1.96 is the ~95% interval.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, {n}], got {k}")

    p_hat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = p_hat + z2 / (2 * n)
    adj = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))

    low = 0.0 if k == 0 else (center - adj) / denom
    high = 1.0 if k == n else (center + adj) / denom
    return low, high
