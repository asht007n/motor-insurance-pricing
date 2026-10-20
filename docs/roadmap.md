# Roadmap

The frequency model (phase 1) is done. Each later phase answers one more
pricing question, has a test for "done", and is scoped so it can ship on its
own.

| Phase | Question | Status |
|---|---|---|
| 1. Frequency GLM | How often will each driver claim? | done |
| 2. Age × bonus-malus | Does fixing the interaction improve the table? | next |
| 3. Severity GLM | How much does each claim cost? | |
| 4. Pure premium and aggregate losses | What should each driver pay, and how bad can a year get? | |
| 5. GBM challenger | Is machine learning worth giving up the table? | |
| 6. Rating calculator app | Can a non-technical user price a driver? | |
| 7. Tableau dashboard | Can a manager see where the tariff is off? | |

---

## 2. Age × bonus-malus

**Why first.** The current age curve dips to 0.58 at 26–29 because, at the
same bonus-malus, younger drivers have a better claims record (see decision
14). Any reviewer will ask about it.

**Try, in order:**
1. A feature for how far a driver's bonus-malus is above the best value
   reachable at their age (100 × 0.95^(years since 18), floored at 50).
2. An interaction between a young-driver flag (under 30) and bonus-malus bands.

**Done when:** on the same five grouped folds, the change improves holdout
deviance in every fold, and the age relativities become roughly monotone. If
it fails in any fold, keep the simpler model and say so.

## 3. Severity GLM

**Model.** Gamma GLM, log link, on individual claims (`claims_clean`), with
fewer factors than frequency: the drivers of how often someone crashes and
how much a crash costs differ.

**Design inputs from the data** (`docs/data_notes.md`):
- 41 claims above €100,000 carry 24.5% of all claim cost, 17.7% of it in
  the part above €100,000. Cap each claim at
  a threshold chosen from a mean-excess plot, fit the GLM on capped amounts,
  and add the excess back as a flat loading (total excess ÷ total exposure).
- 42% of claims sit on five exact amounts (consistent with the flat-rate
  IRSA convention). A Gamma fit is still a reasonable mean model, but check
  holdout A/E by severity band; a two-part model (probability of a repeated
  flat amount, Gamma for the rest)
  is the alternative if the fit is poor.

**Done when:** holdout A/E on claim cost is within ±5% overall and by
predicted-severity quintile, and the threshold choice is plotted and
justified.

## 4. Pure premium and aggregate losses

- Pure premium = frequency × capped severity × (1 + large-loss loading).
- Cross-check against a single Tweedie GLM on cost per policy-year.
- Holdout: A/E on claim cost by decile of pure premium.
- **Monte Carlo of a year's portfolio losses:** draw Poisson claim counts from
  the frequency model and amounts from the severity model, repeat 10,000
  times, and report the distribution of total losses and its 99.5th
  percentile. Same simulation method as the World Cup and UCL forecasting
  projects, applied to an insurer's capital question.

**Done when:** the rating calculator shows a price in euros with its
frequency and severity parts, and the simulation's mean matches the sum of
pure premiums to within simulation error.

## 5. GBM challenger

**Model.** LightGBM with a Poisson objective and log(exposure) as the base
margin, on unbanded raw features, monotone constraint on bonus-malus.

**Fair comparison:**
- Same five grouped folds as the GLM; tune on an inner split of the training
  folds only.
- Mean holdout Poisson deviance and Gini per fold, and the per-fold
  difference against the GLM.
- Double-lift chart: sort holdout policies by GBM prediction ÷ GLM
  prediction, and show which model's predictions track actual claims where
  they disagree most.
- Partial dependence and SHAP interaction values to find what the GLM
  misses; feed the biggest interaction back into the GLM and re-compare.

**Decide in advance:** adopt the GBM only if it beats the GLM in all five
folds by more than the fold-to-fold spread of the GLM itself (Gini range
0.276–0.304 today), and the double-lift shows the GLM mispricing a segment a
business would care about. Otherwise keep the table and use the GBM as a
diagnostic.

## 6. Rating calculator app

Streamlit app that reads only `reports/tables/relativities.csv` and
`reports/metrics.json` (no raw data, nothing to keep secret), takes a
driver's details, and shows the price as a factor-by-factor waterfall.
Deploy on Streamlit Community Cloud and link it at the top of the README.

**Done when:** the app's price for the README's three example drivers
matches `python -m mtpl.rating` exactly (add a test).

## 7. Tableau dashboard

Tableau Public workbook built on the CSVs in `reports/tables/`: A/E by factor
level with the z-score flag, the lift table, and the relativities. One view
a pricing manager could use to spot a mispriced segment.

**Done when:** published on Tableau Public and linked from the README.
