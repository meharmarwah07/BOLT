"""Wilson score interval for a binomial proportion, and an exact McNemar test."""
import math
from math import comb


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


def mcnemar_exact(b, c):
    """Exact (binomial), two-sided McNemar test on the two DISCORDANT
    counts of a paired binary comparison:

        b = # pairs where A succeeded and B failed
        c = # pairs where A failed and B succeeded

    (concordant pairs -- both succeeded or both failed -- carry no
    information for McNemar and are not passed in.) Under the null
    hypothesis the two conditions are equally likely to be the one that
    "wins" a discordant pair, so b | (b+c) ~ Binomial(b+c, 0.5); this
    returns the exact two-sided p-value for that test (not the chi-square
    approximation, which is unreliable at the small b+c this project's
    seed counts tend to produce).

    Returns 1.0 if b == c == 0 (no discordant pairs -- nothing to test).
    """
    if b < 0 or c < 0:
        raise ValueError(f"b and c must be >= 0, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Two-sided exact p-value: sum the probability of every outcome at
    # least as extreme (in either tail) as the observed split, under
    # Binomial(n, 0.5). By symmetry around n/2, this is twice the lower
    # tail P(X <= k) for k < n/2, with the k == n/2 point counted once.
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = 2 * tail
    if n % 2 == 0 and k == n / 2:
        p -= comb(n, k) / (2 ** n)  # the center term was double-counted
    return min(1.0, p)
