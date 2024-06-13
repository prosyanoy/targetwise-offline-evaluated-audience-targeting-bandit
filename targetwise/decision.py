"""Go / no-go decision combining offline policy evaluation, the A/B test and the budget."""
from __future__ import annotations

from dataclasses import dataclass

from .abtest import ABResult
from .ope import Estimate


@dataclass
class Criterion:
    name: str
    passed: bool
    detail: str


@dataclass
class Decision:
    verdict: str  # GO | NO-GO | INCONCLUSIVE
    criteria: list[Criterion]
    rationale: str


def decide(ope_lift: dict[str, Estimate], ab: ABResult, planned_n: int, required_n: int,
           spend: float, budget: float, min_lift: float = 0.0, budget_tol: float = 0.05) -> Decision:
    """``ope_lift`` maps estimator name -> paired lift estimate of the candidate over the control."""
    dr = ope_lift["DR"]
    c_ope = Criterion("OPE: DR lift CI excludes 0", dr.lo > 0,
                      f"lift {dr.value:+.4f} [{dr.lo:+.4f}, {dr.hi:+.4f}]")
    c_agree = Criterion("OPE: IPS, SNIPS, DR all positive", all(e.value > 0 for e in ope_lift.values()),
                        ", ".join(f"{k} {e.value:+.4f}" for k, e in ope_lift.items()))
    c_ab = Criterion("A/B: one-sided lower bound > min lift", ab.lower_one_sided > min_lift,
                     f"lift {ab.lift:+.4f}, lower bound {ab.lower_one_sided:+.4f}, p={ab.p_value:.4g}")
    c_pow = Criterion("A/B: sample size meets power plan", planned_n >= required_n,
                      f"n/arm {planned_n} vs required {required_n}")
    c_bud = Criterion("Budget respected", spend <= budget * (1 + budget_tol),
                      f"spend/user {spend:.4f} vs budget {budget:.4f}")
    criteria = [c_ope, c_agree, c_ab, c_pow, c_bud]

    if not c_bud.passed:
        v, why = "NO-GO", "Realised spend exceeds the budget."
    elif c_ope.passed and c_agree.passed and c_ab.passed:
        v = "GO"
        why = "Offline evaluation and the randomised test agree that the policy beats the control."
        if not c_pow.passed:
            why += " Caveat: the test was smaller than the power plan."
    elif ab.upper_one_sided <= min_lift or dr.hi <= 0:
        v, why = "NO-GO", "Evidence rules out a worthwhile lift."
    else:
        v, why = "INCONCLUSIVE", "Evidence is not strong enough either way; collect more data or revise the policy."
    return Decision(v, criteria, why)
