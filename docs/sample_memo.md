# Launch decision memo: audience targeting policy `thompson+budget`

**Decision: GO**. Launch to the full audience with the budget cap and a holdout for monitoring.

## Proposal
Replace the current targeting rule with a bandit-learned policy that targets users by expected incremental
revenue per unit cost, capped at 0.374 spend per user (30% of the cost of targeting everyone).

## Evidence
- **Offline (counterfactual, 30000 held-out logged users):** doubly robust lift over the current rule is
  +1.23 net value/user (95% CI +1.09 to +1.36); IPS and self-normalised IPS agree in sign and size.
  Versus not targeting at all: +0.52 (+0.38 to +0.66).
- **Randomised A/B test (4575 users per arm):** lift +1.09 (95% CI +0.65 to +1.54),
  one-sided p = 7.8e-07. Spend per user: 0.306 vs 0.654 for the current rule.
- **Power:** the test was sized for 0.61, half of the offline lift, at 80% power and alpha = 0.05; needed 4575 per arm.
- **Why trust the offline numbers:** the estimators were also scored against oracle policy values: across 5 seeds x 4 candidates, the DR lift estimate was within 6% of the oracle online lift (median error 3.0%).

## Decision criteria
- PASS: OPE: DR lift CI excludes 0 (lift +1.2255 [+1.0887, +1.3624])
- PASS: OPE: IPS, SNIPS, DR all positive (IPS +1.2198, SNIPS +1.2158, DR +1.2255)
- PASS: A/B: one-sided lower bound > min lift (lift +1.0947, lower bound +0.7199, p=7.778e-07)
- PASS: A/B: sample size meets power plan (n/arm 4575 vs required 4575)
- PASS: Budget respected (spend/user 0.3064 vs budget 0.3744)

## Risks and mitigations
- **Selection optimism:** the candidate was picked as the best of four on the same log; the fresh-user A/B test is the unbiased confirmation.
- **Distribution shift:** the policy was learned from one logging period. Keep a randomised holdout and re-run offline evaluation monthly.
- **Overlap:** estimates are only valid where the old rule gave every user a non-trivial chance of each action (propensities in [0.1, 0.9]).
- **Simulation caveat:** these numbers come from a simulator with known truth, not production traffic.

## Next steps
1. Roll out to a small share of traffic with the budget cap enforced. 2. Monitor realised spend and lift against the interval above. 3. Re-train and re-evaluate on fresh logs before widening.
