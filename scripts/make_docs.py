"""Run from the repo root: python -m scripts.make_docs
Regenerate docs/estimator_vs_truth.png, docs/sample_memo.md and docs/accuracy.md (needs requirements-docs.txt)."""
from pathlib import Path

from targetwise.accuracy import estimator_accuracy, summarise
from targetwise.config import Config
from targetwise.pipeline import run_pipeline
from targetwise.plots import plot_estimated_vs_true
from targetwise.report import render_memo

docs = Path(__file__).resolve().parent.parent / "docs"
docs.mkdir(exist_ok=True)
cfg = Config()
res = run_pipeline(cfg, write=False)
plot_estimated_vs_true(res, docs / "estimator_vs_truth.png")
acc = estimator_accuracy(cfg)
summ = summarise(acc)
dr = summ.loc["DR"]
line = (f"across {acc.seed.nunique()} seeds x {acc.candidate.nunique()} candidates, the DR lift estimate was within "
        f"{dr.max_rel_err:.0%} of the oracle online lift (median error {dr.median_rel_err:.1%}).")
(docs / "sample_memo.md").write_text(render_memo(res, line))
(docs / "accuracy.md").write_text(
    "| Estimator | Median relative error | Max relative error | 95% CI coverage |\n|---|---|---|---|\n" +
    "\n".join(f"| {k} | {r.median_rel_err:.1%} | {r.max_rel_err:.1%} | {r.ci_coverage:.0%} |" for k, r in summ.iterrows()) + "\n")
print(line)
