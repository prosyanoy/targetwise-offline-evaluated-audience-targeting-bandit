"""Configuration read from environment variables (optionally a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader; existing environment variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@dataclass(frozen=True)
class Config:
    seed: int = 7
    n_logged: int = 60_000
    value_per_conversion: float = 30.0
    budget_frac: float = 0.3  # fraction of the cost of targeting everybody
    alpha: float = 0.05
    power: float = 0.8
    ab_max_n: int = 100_000  # cap on users per A/B arm
    effect_haircut: float = 0.5  # plan the A/B on this share of the OPE lift
    min_lift: float = 0.0  # practical significance threshold (net value/user)
    online_rounds: int = 20_000
    epsilon: float = 0.1
    linucb_alpha: float = 0.5
    output_dir: str = "reports"

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        env_names = {
            "value_per_conversion": "TW_VALUE_PER_CONVERSION",
        }
        kwargs = {}
        for f in fields(cls):
            name = env_names.get(f.name, "TW_" + f.name.upper())
            raw = os.environ.get(name)
            if raw is not None and raw != "":
                kwargs[f.name] = type(f.default)(raw)
        return cls(**kwargs)
