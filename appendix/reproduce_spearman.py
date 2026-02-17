#!/usr/bin/env python3
"""Reproduce basic Spearman correlations reported in Appendix A.

Input:
  appendix/attribute_matrix.csv  (derived from Table 3.D)

No third-party dependencies.

This script converts categorical fields into ordinal scores using the paper's
operational conventions, then prints a Spearman correlation matrix among:
  T = time horizon, E = energy, V = variability, C = concentration.

Notes:
- Range expressions like "Long/Deep" are reduced by taking the max (conservative).
- "Fixed→Fork" is treated as Fork (max variability).
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Iterable


def max_of_tokens(s: str, mapping: dict[str, int]) -> int | None:
    toks = re.split(r"\s*/\s*", s.strip())
    vals: list[int] = []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        for k, v in mapping.items():
            if k in t:
                vals.append(v)
    return max(vals) if vals else None


def parse_energy(s: str) -> int | None:
    m = re.search(r"\((\d+)\)", str(s))
    return int(m.group(1)) if m else None


def rankdata(xs: list[float]) -> list[float]:
    """Average ranks for ties (1-indexed)."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        # average rank among i..j (1-indexed)
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearsonr(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx == 0 or vy == 0:
        return float("nan")
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    return cov / math.sqrt(vx * vy)


def spearmanr(x: list[float], y: list[float]) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    return pearsonr(rx, ry)


def read_scores(csv_path: Path) -> dict[str, list[float]]:
    map_T = {"Ephemeral": 1, "Mid": 2, "Long": 3, "Deep": 4}
    map_V = {"Fixed→Fork": 4, "Fork": 4, "Evolutionary": 3, "Versioned": 2, "Fixed": 1}
    map_C = {"Distributed": 1, "Networked": 2, "Centralized": 4}

    out = {"T": [], "E": [], "V": [], "C": []}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            T = max_of_tokens(row["time_horizon"], map_T)
            V = max_of_tokens(row["variability"], map_V)
            C = map_C.get(row["power_orientation"].strip())
            E = parse_energy(row["energy"])

            if None in (T, V, C, E):
                raise ValueError(f"Could not parse row: {row}")

            out["T"].append(float(T))
            out["V"].append(float(V))
            out["C"].append(float(C))
            out["E"].append(float(E))

    return out


def main() -> None:
    scores = read_scores(Path("appendix/attribute_matrix.csv"))
    keys = ["T", "E", "V", "C"]

    # correlation matrix
    print("# Spearman correlation (rho)")
    header = "      " + " ".join(f"{k:>7}" for k in keys)
    print(header)
    for ki in keys:
        row = [ki]
        for kj in keys:
            rho = 1.0 if ki == kj else spearmanr(scores[ki], scores[kj])
            row.append(f"{rho:7.3f}")
        print(f"{row[0]:>3}  " + " ".join(row[1:]))


if __name__ == "__main__":
    main()
