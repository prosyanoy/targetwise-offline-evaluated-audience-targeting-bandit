"""Optional dashboard: `streamlit run app/streamlit_app.py` (needs `pip install streamlit`)."""
from dataclasses import replace

import pandas as pd
import streamlit as st

from targetwise.config import Config
from targetwise.pipeline import run_pipeline
from targetwise.report import render_report

st.title("TargetWise: offline-evaluated audience targeting")
base = Config.from_env()
budget = st.sidebar.slider("Budget (share of target-everyone cost)", 0.05, 1.0, base.budget_frac, 0.05)
n = st.sidebar.select_slider("Logged users", [20_000, 40_000, 60_000, 100_000], base.n_logged)
seed = st.sidebar.number_input("Seed", value=base.seed, step=1)


@st.cache_data(show_spinner=True)
def run(budget, n, seed):
    return run_pipeline(replace(base, budget_frac=budget, n_logged=n, seed=int(seed), online_rounds=10_000), write=False)


res = run(budget, n, seed)
d = res["decision"]
st.header(f"Decision: {d.verdict}")
st.write(d.rationale)
st.table(pd.DataFrame([{"criterion": c.name, "pass": c.passed, "detail": c.detail} for c in d.criteria]))
st.subheader("OPE value vs oracle truth")
st.dataframe(pd.DataFrame({k: {"DR": v["DR"].value, "DR lo": v["DR"].lo, "DR hi": v["DR"].hi,
                               "truth": res["truth"][k]} for k, v in res["estimates"].items()}).T)
st.subheader("Online cumulative regret")
st.line_chart(pd.DataFrame(res["online_regret"]))
with st.expander("Full markdown report"):
    st.markdown(render_report(res))
