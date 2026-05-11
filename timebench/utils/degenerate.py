"""Detect and collapse runaway repetition loops in LLM output text.

The public entry point is :func:`find_endless_repetition`, which scans the
tail of a string for consecutive repeated substrings ("loops") and collapses
them to a single copy of the repeating unit.
"""

from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Internal polynomial-hash helpers (double-hash to reduce collisions)
# ---------------------------------------------------------------------------

_MASK64: int = (1 << 64) - 1
_BASE1: int = 1_315_423_911
_BASE2: int = 2_654_435_761


def _build_hash64(s: str, base: int) -> Tuple[List[int], List[int]]:
    """Build prefix-hash and prefix-power tables for *s* using *base*."""
    n = len(s)
    h = [0] * (n + 1)
    p = [1] * (n + 1)
    for i, ch in enumerate(s, 1):
        h[i] = (h[i - 1] * base + ord(ch)) & _MASK64
        p[i] = (p[i - 1] * base) & _MASK64
    return h, p


def _subhash64(h: List[int], p: List[int], l: int, r: int) -> int:
    """Return the rolling hash of ``s[l:r]`` given pre-built tables *h* and *p*."""
    return (h[r] - (h[l] * p[r - l] & _MASK64)) & _MASK64


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_endless_repetition(
    s: str,
    *,
    min_repeats: int = 4,
    max_unit: int = 2048,
    lookback: int = 8000,
    max_trim: int = 512,
) -> Tuple[bool, str]:
    """Detect a consecutive repeated substring near the end of *s* and fix it.

    Returns ``(is_repetitive, fixed_text)``.

    * Operates on raw characters — no normalisation is applied.
    * Keeps exactly **one** copy of the repeating unit and removes the rest.
    * Handles a cut-off mid-repeat by trimming back to the nearest clean
      loop boundary.

    Parameters
    ----------
    s:
        The string to inspect.
    min_repeats:
        Minimum number of consecutive repetitions required to trigger detection.
        Very short units (≤16 chars) require proportionally more repeats.
    max_unit:
        Maximum length (in characters) of a candidate repeating unit.
    lookback:
        Only analyse the last *lookback* characters (performance guard).
    max_trim:
        How far back from the tail we consider when searching for a clean
        loop boundary.
    """
    n = len(s)
    if n < 8:
        return False, s

    tail_start = max(0, n - lookback)
    tail = s[tail_start:]
    L = len(tail)
    if L < 2:
        return False, s

    # Candidate end-positions to try, ordered strong → soft punctuation
    strong_end = set(".!?\n\r")
    soft_end = set(" \t,;:-—()[]{}\"'""''/\\|")
    lo = max(1, L - max_trim)

    punct_ends: List[int] = []
    soft_ends: List[int] = []
    for i in range(L, lo, -1):
        ch = tail[i - 1]
        if ch in strong_end:
            punct_ends.append(i)
        elif ch in soft_end:
            soft_ends.append(i)

    ends = punct_ends + soft_ends

    # Deduplicate while preserving order
    seen: set = set()
    ends = [e for e in ends if not (e in seen or seen.add(e))]  # type: ignore[func-returns-value]

    # Always include the true tail end; prefer strong-punctuation ends
    if (tail[-1] in strong_end) or (not ends):
        ends = [L] + ends

    # Build double hash tables over the tail
    h1, p1 = _build_hash64(tail, _BASE1)
    h2, p2 = _build_hash64(tail, _BASE2)

    def sub2(l: int, r: int) -> tuple:
        return (_subhash64(h1, p1, l, r), _subhash64(h2, p2, l, r))

    best: Optional[tuple] = None

    for end in ends:
        max_p = min(max_unit, end // min_repeats)
        if max_p < 1:
            continue

        for p in range(1, max_p + 1):
            if end - 2 * p < 0:
                break

            unit_hash = sub2(end - p, end)
            if unit_hash != sub2(end - 2 * p, end - p):
                continue

            # Count how many consecutive repetitions of length p end at `end`
            k = 2
            while (
                end - (k + 1) * p >= 0
                and sub2(end - (k + 1) * p, end - k * p) == unit_hash
            ):
                k += 1

            # Shorter units require more repetitions to avoid false positives
            need = min_repeats
            if p <= 2:
                need = max(12, need)
            elif p <= 4:
                need = max(10, need)
            elif p <= 8:
                need = max(8, need)
            elif p <= 16:
                need = max(6, need)

            if k < need:
                continue

            total = k * p
            trim = L - end
            unit = tail[end - p:end]
            remainder = tail[end:]
            partial = trim > 0 and unit.startswith(remainder)
            trim_pref = trim if partial else -trim
            key = (total, 1 if partial else 0, trim_pref, k, -p)

            if best is None or key > best[0]:
                start_in_tail = end - total
                start_global = tail_start + start_in_tail
                end_global = tail_start + end
                unit_start_global = tail_start + (end - p)
                unit_end_global = tail_start + end
                best = (key, start_global, end_global, unit_start_global, unit_end_global)

    if best is None:
        return False, s

    _, start_g, end_g, unit_s, unit_e = best
    initial = s[:start_g]
    unit = s[unit_s:unit_e]

    if unit.strip():
        check = initial.rfind(unit)
        while check != -1 or (
            len(initial) - 2 * len(unit) - check < 4
            and len(initial) - 2 * len(unit) - check > -4
        ):
            initial = initial[:check]
            check = initial.rfind(unit)

        return True, initial + unit

    return False, s
