"""Markdown go/no-go report."""
from __future__ import annotations


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _ci(e) -> str:
    return f"{e.value:+.3f} [{e.lo:+.3f}, {e.hi:+.3f}]"


def render_report(res: dict) -> str:
    cfg, d, ab, pw = res["config"], res["decision"], res["ab"], res["power"]
    w = res["winner"]
    L = [f"# TargetWise go/no-go report\n", f"## Decision: **{d.verdict}**\n", f"{d.rationale}\n",
         f"Candidate policy: `{w}` vs control `logging_policy` (the current targeting rule).\n",
         table(["Criterion", "Result", "Detail"],
               [[c.name, "PASS" if c.passed else "FAIL", c.detail] for c in d.criteria]), "",
         "## Data\n",
         f"- Logged users: {res['n_train']} train / {res['n_eval']} held-out evaluation (known, confounded logging policy)",
         f"- Naive treated-vs-untreated revenue difference: {res['naive_ate']:+.3f} (biased); true average uplift: {res['true_ate']:+.3f}",
         f"- Budget: {res['budget_per_user']:.3f} per user ({cfg.budget_frac:.0%} of the cost of targeting everyone)\n",
         "## Offline policy evaluation (net value per user, 95% CI)\n"]
    rows = []
    for name, e in res["estimates"].items():
        rows.append([name, _ci(e["IPS"]), _ci(e["SNIPS"]), _ci(e["DR"]), f"{res['truth'][name]:+.3f}",
                     f"{res['spend'][name]:.3f}", f"{e['IPS'].ess:.0f}"])
    L += [table(["Policy", "IPS", "SNIPS", "DR", "True value (oracle)", "Spend/user", "ESS"], rows), "",
          "## Lift over the logging policy (paired, budget-constrained candidates)\n",
          table(["Candidate", "IPS", "SNIPS", "DR"],
                [[n, _ci(v["IPS"]), _ci(v["SNIPS"]), _ci(v["DR"])] for n, v in res["lift"].items()]), "",
          f"Winner's DR lift over *no targeting at all*: {_ci(res['lift_vs_no_targeting'])}\n",
          "## Power analysis\n",
          f"- Outcome sd (net value/user): {pw['sd']:.2f}; planned effect (haircut OPE lift): {pw['plan_effect']:.4f}",
          f"- Required n per arm for {cfg.power:.0%} power at alpha={cfg.alpha}: {pw['required_n']}; used: {pw['n_used']}",
          f"- Analytic power at planned effect: {pw['analytic_power']:.2f}; simulated power at the policy's true effect (n={pw['empirical_power_n']}, 100 runs): {pw['empirical_power']:.2f}",
          f"- Minimum detectable effect at used n: {pw['mde_at_n']:.4f}\n",
          "## Simulated A/B test\n",
          table(["Arm", "Mean net value/user", "Spend/user"],
                [["treatment", f"{ab.mean_treat:.4f}", f"{ab.spend_treat:.4f}"],
                 ["control", f"{ab.mean_control:.4f}", f"{ab.spend_control:.4f}"]]), "",
          f"Lift {ab.lift:+.4f} (SE {ab.se:.4f}), 95% CI [{ab.ci_two_sided[0]:+.4f}, {ab.ci_two_sided[1]:+.4f}], one-sided p = {ab.p_value:.4g}, n = {ab.n_per_arm} per arm.\n",
          "## Online learning regret (from-scratch bandits, cumulative pseudo-regret)\n"]
    names = list(res["online_regret"])
    n_pts = len(res["online_regret"][names[0]])
    idx = sorted({n_pts // 4 - 1, n_pts // 2 - 1, n_pts - 1})
    L.append(table(["Rounds"] + names, [[str((i + 1) * 500)] + [f"{res['online_regret'][n][i]:.0f}" for n in names] for i in idx]))
    return "\n".join(L) + "\n"


def render_memo(res: dict, accuracy_line: str = "") -> str:
    """One-page launch decision memo built from a pipeline result."""
    cfg, d, ab, pw, w = res["config"], res["decision"], res["ab"], res["power"], res["winner"]
    dr = res["lift"][w]["DR"]
    none = res["lift_vs_no_targeting"]
    rec = {"GO": "Launch to the full audience with the budget cap and a holdout for monitoring.",
           "NO-GO": "Do not launch. Keep the current rule and revisit the policy or the budget.",
           "INCONCLUSIVE": "Do not launch yet. Extend the test or improve the policy, then re-run this review."}[d.verdict]
    crit = "\n".join(f"- {'PASS' if c.passed else 'FAIL'}: {c.name} ({c.detail})" for c in d.criteria)
    return f"""# Launch decision memo: audience targeting policy `{w}`

**Decision: {d.verdict}**. {rec}

## Proposal
Replace the current targeting rule with a bandit-learned policy that targets users by expected incremental
revenue per unit cost, capped at {res['budget_per_user']:.3f} spend per user ({cfg.budget_frac:.0%} of the cost of targeting everyone).

## Evidence
- **Offline (counterfactual, {res['n_eval']} held-out logged users):** doubly robust lift over the current rule is
  {dr.value:+.2f} net value/user (95% CI {dr.lo:+.2f} to {dr.hi:+.2f}); IPS and self-normalised IPS agree in sign and size.
  Versus not targeting at all: {none.value:+.2f} ({none.lo:+.2f} to {none.hi:+.2f}).
- **Randomised A/B test ({ab.n_per_arm} users per arm):** lift {ab.lift:+.2f} (95% CI {ab.ci_two_sided[0]:+.2f} to {ab.ci_two_sided[1]:+.2f}),
  one-sided p = {ab.p_value:.2g}. Spend per user: {ab.spend_treat:.3f} vs {ab.spend_control:.3f} for the current rule.
- **Power:** the test was sized for {pw['plan_effect']:.2f}, half of the offline lift, at {cfg.power:.0%} power and alpha = {cfg.alpha}; needed {pw['required_n']} per arm.
- **Why trust the offline numbers:** the estimators were also scored against oracle policy values: {accuracy_line or 'see README'}

## Decision criteria
{crit}

## Risks and mitigations
- **Selection optimism:** the candidate was picked as the best of four on the same log; the fresh-user A/B test is the unbiased confirmation.
- **Distribution shift:** the policy was learned from one logging period. Keep a randomised holdout and re-run offline evaluation monthly.
- **Overlap:** estimates are only valid where the old rule gave every user a non-trivial chance of each action (propensities in [0.1, 0.9]).
- **Simulation caveat:** these numbers come from a simulator with known truth, not production traffic.

## Next steps
1. Roll out to a small share of traffic with the budget cap enforced. 2. Monitor realised spend and lift against the interval above. 3. Re-train and re-evaluate on fresh logs before widening.
"""
