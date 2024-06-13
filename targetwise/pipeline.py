"""End-to-end run: log -> learn -> offline evaluate -> pick -> power plan -> A/B -> go/no-go."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ope
from .abtest import achieved_power, empirical_power, minimum_detectable_effect, run_ab_test, sample_size_per_arm
from .budget import BudgetedTargeter
from .config import Config
from .decision import decide
from .learners import EpsilonGreedy, LinUCB, ThompsonSampling, simulate_online
from .report import render_report
from .simulator import FixedPolicy, LoggingPolicy, Simulator
from .uplift import TwoModelUplift


def run_pipeline(cfg: Config | None = None, write: bool = True) -> dict:
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.seed)
    sim = Simulator(cfg.value_per_conversion)
    scale = cfg.value_per_conversion * 0.3

    # 1. logged data from the known (confounded) logging policy, split train / evaluation
    data = sim.generate_logged(cfg.n_logged, rng)
    train, ev = data.split(0.5, rng)
    naive = float(data.rev[data.action == 1].mean() - data.rev[data.action == 0].mean())
    true_ate = float(sim.true_uplift(sim.sample_context(100_000, rng)[0]).mean())

    # 2. learners (logged-data replay with IPS weights) + two-model uplift baseline
    scorers = {
        "epsilon_greedy": EpsilonGreedy(cfg.epsilon, reward_scale=scale, seed=cfg.seed).fit_logged(train),
        "linucb": LinUCB(cfg.linucb_alpha, reward_scale=scale, seed=cfg.seed).fit_logged(train),
        "thompson": ThompsonSampling(reward_scale=scale, seed=cfg.seed).fit_logged(train),
        "uplift_two_model": TwoModelUplift(cfg.value_per_conversion, cfg.seed).fit(train),
    }
    q_model = TwoModelUplift(cfg.value_per_conversion, cfg.seed + 1).fit(train)
    q0, q1 = q_model.mean_rev(ev.X, 0), q_model.mean_rev(ev.X, 1)

    # 3. policies: baselines, unconstrained learners, budget-constrained learners
    budget = cfg.budget_frac * float(train.cost.mean())
    logging = LoggingPolicy(sim)
    policies = {
        "no_targeting": FixedPolicy(0.0, "no_targeting"),
        "target_all": FixedPolicy(1.0, "target_all"),
        "logging_policy": logging,
    }
    for name, s in scorers.items():
        policies[name] = s
    budgeted = {}
    for name, s in scorers.items():
        b = BudgetedTargeter(s, budget).calibrate(ev.X, ev.cost)  # contexts only, no outcomes used
        budgeted[b.name] = b
        policies[b.name] = b

    # 4. offline policy evaluation on held-out log, compared against ground truth
    est, truth, spend = {}, {}, {}
    for name, p in policies.items():
        pi1 = p.prob_treat(ev.X, ev.cost)
        est[name] = ope.evaluate_all(ev, pi1, q0, q1, cfg.alpha)
        truth[name] = sim.true_policy_value(p.prob_treat, np.random.default_rng(cfg.seed + 99))
        spend[name] = float((pi1 * ev.cost).mean())
    lift = {n: {k: ope.difference(est[n][k], est["logging_policy"][k], cfg.alpha) for k in est[n]}
            for n in budgeted}

    # 5. choose the candidate with the best DR lower bound of lift over the logging policy
    winner = max(budgeted, key=lambda n: lift[n]["DR"].lo)
    vs_none = ope.difference(est[winner]["DR"], est["no_targeting"]["DR"], cfg.alpha)
    cand = budgeted[winner]

    # 6. power analysis and simulated A/B test against the current (logging) policy
    sd = float(ev.reward.std(ddof=1))
    point = lift[winner]["DR"].value
    plan_effect = max(cfg.effect_haircut * point, 0.02)
    required_n = sample_size_per_arm(plan_effect, sd, cfg.alpha, cfg.power)
    n_ab = min(required_n, cfg.ab_max_n)
    ab_rng = np.random.default_rng(cfg.seed + 7)
    ab = run_ab_test(sim, cand, logging, n_ab, ab_rng, cfg.alpha)
    power = {
        "plan_effect": plan_effect, "sd": sd, "required_n": required_n, "n_used": n_ab,
        "mde_at_n": minimum_detectable_effect(n_ab, sd, cfg.alpha, cfg.power),
        "analytic_power": achieved_power(plan_effect, sd, n_ab, cfg.alpha),
        "empirical_power": empirical_power(sim, cand, logging, min(n_ab, 20_000), 100, ab_rng, cfg.alpha)
        if point > 0 else float("nan"),
        "empirical_power_n": min(n_ab, 20_000),
    }

    # 7. go / no-go
    decision = decide(lift[winner], ab, n_ab, required_n, ab.spend_treat, budget, cfg.min_lift)

    # 8. online (from-scratch) bandit regret curves, for context
    online = {}
    for name, cls, kw in [("epsilon_greedy", EpsilonGreedy, {"epsilon": cfg.epsilon}),
                          ("linucb", LinUCB, {"alpha": cfg.linucb_alpha}), ("thompson", ThompsonSampling, {})]:
        r = simulate_online(cls(reward_scale=scale, seed=cfg.seed, **kw), sim, cfg.online_rounds, 500,
                            np.random.default_rng(cfg.seed + 5))
        online[name] = r.tolist()

    res = {
        "config": cfg, "n_train": train.n, "n_eval": ev.n, "naive_ate": naive, "true_ate": true_ate,
        "budget_per_user": budget, "estimates": est, "truth": truth, "spend": spend, "lift": lift,
        "winner": winner, "lift_vs_no_targeting": vs_none, "ab": ab, "power": power, "decision": decision, "online_regret": online,
    }
    if write:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(render_report(res))
        (out / "results.json").write_text(json.dumps(_jsonable(res), indent=2, default=_default))
    return res


def _default(o):
    if hasattr(o, "__dataclass_fields__"):
        return {k: v for k, v in o.__dict__.items() if k != "psi"}
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _jsonable(res):
    return json.loads(json.dumps(res, default=_default))


def main() -> None:
    res = run_pipeline(Config.from_env())
    out = Path(res["config"].output_dir)
    print(render_report(res))
    print(f"\nWrote {out / 'report.md'} and {out / 'results.json'}")


if __name__ == "__main__":
    main()
