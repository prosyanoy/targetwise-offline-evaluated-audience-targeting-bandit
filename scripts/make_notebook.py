"""Run from the repo root: python -m scripts.make_notebook  (writes notebooks/demo.ipynb, unexecuted)."""
import json
from pathlib import Path

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip().splitlines(True)}

cells = [
    md("# TargetWise demo\n\nLog -> learn -> evaluate offline -> A/B -> go/no-go. Run all cells; the last one prints the launch report."),
    code("from dataclasses import replace\nimport io\n\nfrom IPython.display import Image, Markdown, display\n\nfrom targetwise.config import Config\n"
         "from targetwise.pipeline import run_pipeline\nfrom targetwise.plots import plot_estimated_vs_true\n"
         "from targetwise.report import render_memo, render_report\n\ndef show(fig):\n    buf = io.BytesIO()\n    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')\n    display(Image(buf.getvalue()))\n\n\ncfg = replace(Config.from_env(), output_dir='reports')\n"
         "res = run_pipeline(cfg, write=False)  # simulate logs, train 4 learners, OPE, budget, A/B, decision"),
    md("## 1. Logged data is confounded\nThe naive treated-vs-untreated difference is biased; the true average uplift is tiny."),
    code("print(f\"naive difference: {res['naive_ate']:+.3f}   true average uplift: {res['true_ate']:+.3f}\")"),
    md("## 2. Offline estimates vs ground truth\nEach dot is a policy. Left: estimate against the oracle value (on the dashed line = perfect). Right: the error with its 95% CI."),
    code("show(plot_estimated_vs_true(res))"),
    md("## 3. Online learning regret\nCumulative pseudo-regret of the from-scratch bandits on a fresh stream."),
    code("import pandas as pd\nshow(pd.DataFrame(res['online_regret'], index=[(i + 1) * 500 for i in range(len(res['online_regret']['linucb']))]).plot(\n"
         "    xlabel='rounds', ylabel='cumulative regret', figsize=(6, 3)).figure)"),
    md("## 4. Launch decision memo and go/no-go report"),
    code("display(Markdown(render_memo(res)))"),
    code("display(Markdown(render_report(res)))\nprint('DECISION:', res['decision'].verdict)"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent.parent / "notebooks" / "demo.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
