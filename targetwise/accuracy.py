"""How close are offline estimates to the oracle online result? (multi-seed check)"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import Config
from .pipeline import run_pipeline


def estimator_accuracy(cfg: Config, seeds=range(5)) -> pd.DataFrame:
    """Per seed x budgeted candidate x estimator: estimated lift over the logging policy vs the oracle lift."""
    rows = []
    for s in seeds:
        res = run_pipeline(replace(cfg, seed=cfg.seed + s, online_rounds=500, ab_max_n=2000), write=False)
        for cand, ests in res["lift"].items():
            true = res["truth"][cand] - res["truth"]["logging_policy"]
            for k, e in ests.items():
                rows.append({"seed": cfg.seed + s, "candidate": cand, "estimator": k, "estimated": e.value,
                             "true": true, "abs_err": abs(e.value - true), "rel_err": abs(e.value - true) / abs(true),
                             "covered": e.lo <= true <= e.hi})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("estimator")
    return pd.DataFrame({"median_rel_err": g.rel_err.median(), "max_rel_err": g.rel_err.max(),
                         "ci_coverage": g.covered.mean()})
