import json
from dataclasses import replace

from targetwise.config import Config
from targetwise.pipeline import run_pipeline


def test_pipeline_end_to_end(tmp_path):
    cfg = Config(n_logged=20_000, online_rounds=2000, output_dir=str(tmp_path), ab_max_n=20_000)
    res = run_pipeline(cfg)
    assert res["decision"].verdict in {"GO", "NO-GO", "INCONCLUSIVE"}
    assert res["spend"][res["winner"]] <= res["budget_per_user"] * 1.1
    # OPE estimates agree with oracle truth within CI-ish tolerance for the logging policy
    e = res["estimates"]["logging_policy"]["DR"]
    assert abs(e.value - res["truth"]["logging_policy"]) < 4 * e.se + 0.1
    assert "Decision" in (tmp_path / "report.md").read_text()
    assert "estimates" in json.loads((tmp_path / "results.json").read_text())


def test_deterministic_given_seed(tmp_path):
    cfg = Config(n_logged=10_000, online_rounds=1000, output_dir=str(tmp_path), ab_max_n=5000)
    a, b = run_pipeline(cfg, write=False), run_pipeline(replace(cfg), write=False)
    assert a["ab"].lift == b["ab"].lift and a["winner"] == b["winner"]


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("TW_SEED", "3")
    monkeypatch.setenv("TW_BUDGET_FRAC", "0.5")
    c = Config.from_env()
    assert c.seed == 3 and c.budget_frac == 0.5


def test_memo_and_chart(tmp_path):
    import pytest

    from targetwise.report import render_memo

    res = run_pipeline(Config(n_logged=10_000, online_rounds=1000, ab_max_n=5000), write=False)
    memo = render_memo(res, "within X%")
    assert res["decision"].verdict in memo and "Risks and mitigations" in memo and "within X%" in memo
    pytest.importorskip("matplotlib")
    from targetwise.plots import plot_estimated_vs_true

    out = tmp_path / "c.png"
    plot_estimated_vs_true(res, out)
    assert out.stat().st_size > 1000
