# Modeling decisions

Every choice in the model, why it was made, what else was considered, and
what in this data supports it. Numbers come from `reports/metrics.json` and
`reports/tables/`.

---

## 1. The severity table decides the claim count

**Decision.** For every policy, `ClaimNb` is set to the number of claim
records in `freMTPL2sev`. Claim records with no matching policy are dropped.

**Why.** A pure premium is frequency × severity. If the frequency model counts
claims that have no amount, the two halves describe different claims and the
premium will not add up to the losses. The severity table is the side that
carries money, so it wins.

**Alternatives.** Trust the policy table's count and give unmatched claims an
average amount; or drop mismatched policies entirely. The first invents data,
the second throws away exposure and biases frequency down.

**Evidence here.** In the pinned CASdatasets version every one of 677,991
policies already matches (0 changes, 0 orphans). The rule matters for the
older OpenML/Kaggle copies, where about 9,000 policies are reported not to
match, and it is tested on a toy database with each kind of mismatch.

---

## 2. Cap claim counts at 4

**Decision.** `ClaimNb = min(ClaimNb, 4)` for the frequency model. Claim
amounts stay uncapped in the claims table for the severity model.

**Why.** Eight policies have 5–16 claims. Seven of them share a near-identical
profile (age 52–53, bonus-malus 50, brand B1, Languedoc-Roussillon) and were
insured for 0.07–0.41 years: 66 claims on about 1.4 policy-years. That looks
like one fleet or duplicated records, not seven drivers. Under a log link
these rows would pull the B1 and Languedoc-Roussillon relativities for every
other driver.

**Cost.** 39 claims removed of 26,444 (0.15%).

**Alternatives.** Drop the eight policies (loses the information that they did
claim); keep them (lets one suspect cluster set two relativities). The
scikit-learn freMTPL2 example uses the same cap.

---

## 3. Cap exposure at 1 year

**Decision.** `Exposure = min(Exposure, 1)`. Affects 1,224 policies (max 2.01).

**Why.** The data is documented as observed mostly over one year. Exposure
above one year is more likely a recording error than a two-year policy, and
the cap changes total exposure by only 0.04%.

**Honest caveat.** If some of these really were two-year policies, capping
overstates their claim rate slightly. The effect on the relativities has not
been tested separately; it is listed as a limitation.

---

## 4. Poisson GLM with a log link

**Decision.** Claim counts ~ Poisson, log E[claims] = log(exposure) + β₀ + Σ β.

**Why Poisson.** Claim counts are non-negative integers, mostly 0 or 1, and
their variance grows with their mean. Poisson is the natural starting
distribution for counts, and its estimates of the mean stay consistent even
when the variance assumption is off (see decision 11).

**Why a log link.** It makes the model multiplicative:
`rate = base × age factor × bonus-malus factor × ...`. That is exactly the
shape of a rating table, so each exp(β) is a relativity a pricing team can
read and file. It also keeps every predicted rate positive.

**Alternatives.** Linear regression on claim rates (can predict negative
rates, treats a 0 on a one-week policy like a 0 on a full year, wrong variance
for counts). Logistic regression on "had a claim" (throws away the
difference between one and two claims and ignores exposure unless you add
a complementary log-log link, at which point you are close to Poisson anyway).

---

## 5. log(exposure) as an offset

**Decision.** log(exposure) enters with its coefficient fixed at 1.

**Why.** It encodes the assumption that claims accrue in proportion to time
insured: a policy held for three months is expected to have a quarter of a
full year's claims. The model then estimates claims per policy-year. Without
it, short policies look safe simply because they had less time to claim.

**Why fixed at 1 rather than estimated.** Estimating it would let the model
say "doubling exposure gives 1.7× the claims", which is not how a tariff
works (a six-month policy pays roughly half the annual premium). Fitting with
an offset is mathematically the same as modelling claims/exposure with
exposure as the weight.

---

## 6. Keep identical risk profiles on the same side of the split

**Decision.** Holdout of 20%, assigned by risk profile (all nine rating
fields), not by individual policy. Same for the 5-fold cross-validation.

**Why.** 257,908 policies (38%) share all nine fields with another policy.
In 79% of those groups the policy IDs sit within 100 of each other (median
gap 2), and these policies average 0.33 years of exposure against 0.53
overall. Most look like one driver's year split across several records. A
policy-level random split would put part of the same driver in training and
part in the holdout, so the holdout would partly test memory instead of
prediction. A banded GLM cannot memorise individual profiles, so for it the
effect should be small, but a gradient-boosted challenger can, and it must be
judged on the same honest split.

**Limitation.** Unrelated drivers who happen to match on all nine fields are
grouped (harmless). True duplicates that differ on one field are not caught.

**Better, if the data allowed:** an out-of-time split (train on earlier
years, test on later). freMTPL2 has no dates.

---

## 7. Band the continuous factors

**Decision.** Driver age, bonus-malus, vehicle age, power and density are cut
into 7–10 bands each (edges in `config.py`).

**Why.** The effects are not straight lines: claim frequency falls steeply
from age 18 to 30, is flat through middle age and drifts at older ages.
Bands capture any shape with a relativity per band that an underwriter can
read off a table. Edges were chosen from the one-way tables so each band has
enough exposure for a stable estimate (the smallest band, bonus-malus 125+,
still has about 800 policy-years and 360 claims) and the young-driver slope
keeps its own bands.

**Alternatives.** Splines or GAMs (smoother and usually a little more
predictive, but the output is a curve, not a table), or raw continuous terms
(forces a straight line on the log scale). Smoothing the bands is a
reasonable next step.

**Density on bands of roughly equal log width**, because frequency rises
roughly linearly in log(density), not in density.

---

## 8. Leave Area out, keep Density

**Decision.** Area (A–F) is not in the model.

**Why.** Area is a coarse banding of population density: the correlation
between the area code and log density is 0.97. With both in the model, one
effect gets split arbitrarily between two aliased factors, the standard
errors grow, and neither relativity means anything on its own. Density is
the finer of the two, so it stays.

---

## 9. Base level = the level with the most exposure

**Decision.** For each factor, the reference level (relativity 1.00) is its
largest-exposure level: age 40–49, bonus-malus 50, vehicle age 6–9, power 6,
brand B1, regular fuel, density 50–149, region Centre.

**Why.** Predictions do not depend on the choice, but the standard errors of
the relativities do: every relativity is measured against the base, so a
large base makes all of them more precise. It also makes the base driver a
typical one, so the base frequency (0.0485 claims per year) is meaningful.

---

## 10. Fit on rating cells, not individual policies

**Decision.** Policies with identical rating factors are pooled into 193,242
cells (summed claims, summed exposure) before fitting.

**Why.** For a Poisson model the log-likelihood depends on a group of
policies with the same rate only through their total claims and total
exposure. So the cell fit gives exactly the same coefficients and standard
errors as the policy-level fit (a unit test checks that the coefficients
agree to 1e-8) on a
third of the rows. With Newton's method, a run without the negative
binomial check stays near 1.5 GB of memory.

**What does not pool.** Dispersion is a statement about policy-level
variation, so the dispersion checks run on individual policies.

---

## 11. Overdispersion: check it, quantify it, then decide

**Decision.** Keep the Poisson mean model. Inflate standard errors by √φ
(quasi-Poisson), with φ the Pearson dispersion.

**Why this is the right question.** Overdispersion (variance above the mean)
does not bias the Poisson relativities: the Poisson estimating equations only
require the mean to be right. What it does is make the model's standard
errors too small. So the questions are (a) how big is it, and (b) would a
different variance assumption change the relativities?

**Evidence here.**
- Pearson φ = 1.72. But it is fragile: the 100 most extreme of 542,055
  training policies supply 22% of the statistic, and it falls to 1.33
  without the 33,830 policies insured for under 0.05 years. A claim on a
  policy insured for a few days has a tiny expected value and a huge
  (y − μ)²/μ.
- The deviance-based estimate (0.24) is not meaningful for counts this small,
  which is why Pearson is used.
- Cameron–Trivedi regression test: α = 0.67, t = 1.62, not significant at 5%.
- Negative binomial (NB2) refit with that α: no relativity moves more than
  1.1% (median 0.15%), and holdout deviance explained is 5.24% for both.

**Conclusion.** Mild, fragile overdispersion that does not change the
tariff. Scaling the standard errors by √1.72 is the conservative way to
report the uncertainty. "Switch to negative binomial" would have been
defensible too; the evidence says it makes no practical difference.

---

## 12. How the model is judged

| Metric | What it answers | Why not something else |
|---|---|---|
| Poisson deviance | Is the predicted mean close to the counts, using the model's own loss? | RMSE and R² treat a miss on a one-week policy like a miss on a full year and assume constant variance |
| Gini (ordered Lorenz) | Does the model put the risky policies at the top? | AUC ignores exposure and claim counts above 1 |
| Lift / A/E by decile | Is it calibrated where it matters, at the cheap and expensive ends? | One overall A/E can hide offsetting errors |
| A/E by factor level | Is any single rating cell mispriced on new data? | |

Gini here sorts policies from lowest to highest predicted rate, plots
cumulative exposure against cumulative claims, and is 1 − 2 × the area under
that curve. Policies with the same predicted rate are pooled into one step,
because a banded GLM produces many ties and their order would otherwise be
arbitrary. Gini measures ranking only: doubling every prediction leaves it
unchanged, which is why A/E is reported alongside it.

**Why deviance explained is only 5%.** 96% of policies have no claim in the
year. At the level of a single policy, whether a claim happens is mostly
chance. A model can be excellent at telling 0.03-rate drivers from 0.19-rate
drivers (the bottom and top deciles here) and still explain only a few
percent of the deviance.

**A built-in check.** With a log link and an intercept, a Poisson GLM
reproduces total claims exactly on the training data, and does the same
within every level of every factor (the score equation for each 0/1 column
forces it). Training A/E is therefore 1.000 everywhere by construction, so
only holdout A/E is evidence of anything.

---

## 13. Report a range, not one holdout

**Decision.** Besides the fixed holdout, refit on five grouped folds.

**Why.** One split is one draw. Across five folds the Gini ranges from 0.276
to 0.304 (mean 0.295) and deviance explained from 4.3% to 5.1% (mean 4.9%).
The fixed holdout (Gini 0.305, 5.2%) happens to sit just above that range,
so the cross-validation mean is the number to quote. When a challenger model is
compared, a difference smaller than this spread is not evidence.

---

## 14. Keep bonus-malus despite the questions about it

**Decision.** Bonus-malus is in the model.

**Why.** It is the strongest rating factor by far (up to 9.91×) and real
French tariffs use it.

**Two things to say about it.**
1. *Timing.* If the coefficient was recorded at the end of the policy year,
   it partly contains the claims being predicted, and its relativities are
   overstated. The documentation does not say, and the data cannot settle it:
   an off-ladder value (anything other than 50, 51, 54, 57, 60, 64, 68, 72,
   76, 80, 85, 90, 95, 100, which covers 91% of exposure) usually means an at-fault
   claim at some point, but it stays off the ladder for years afterwards, so
   it cannot be tied to the current period. The question I would ask the
   data owner first.
2. *Interaction with age.* A young driver cannot have reached 50 yet: the
   coefficient starts at 100 and falls 5% per claim-free year. So at the same
   bonus-malus, a younger driver tends to have a better claims record than an
   older one. The multiplicative model can only express that through the age
   factor, which is why the age curve dips to 0.58 at 26–29. An age ×
   bonus-malus interaction, or a feature such as "bonus-malus above the best
   value reachable at this age", is the first improvement on the roadmap.
