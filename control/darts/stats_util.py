"""Shared statistics helpers for DART/DART-S paper pipelines."""
from __future__ import annotations

import statistics
from math import comb
from typing import Iterable, Optional


def median_or_none(values: Iterable[float | int | None]) -> Optional[float]:
    """Return the conventional sample median, dropping ``None`` values."""
    clean = [float(v) for v in values if v is not None]
    return float(statistics.median(clean)) if clean else None


def exact_unit_signflip_p(values: Iterable[int | float]) -> float:
    """Exact two-sided sign-flip p-value for contrasts in ``{-1, 0, +1}``.

    Zero contrasts carry no sign information and are removed.  This avoids
    Monte-Carlo jitter in the paper's binary factorial interaction tests.
    """
    clean = [int(value) for value in values if float(value) != 0.0]
    if any(value not in (-1, 1) for value in clean):
        raise ValueError("exact_unit_signflip_p requires {-1, 0, +1} contrasts")
    n = len(clean)
    if n == 0:
        return 1.0
    observed = abs(sum(clean))
    extreme = sum(
        comb(n, k)
        for k in range(n + 1)
        if abs(2 * k - n) >= observed
    )
    return min(1.0, extreme / (2 ** n))


__all__ = ["median_or_none", "exact_unit_signflip_p"]
