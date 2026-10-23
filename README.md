# Who works without a pension in Paraguay?

Six in ten private-sector employees in Paraguay work without contributing to a
pension fund. This asks who they are, how confidently we can say so, and how
much of it is predictable — using the household survey microdata the way a
clustered probability sample has to be used.

**→ [Open the live report](https://sjalegre.github.io/paraguay-informality/)**

Built from the INE's **Encuesta Permanente de Hogares Continua**: 57,744 person
records, 1,693 primary sampling units, expansion factors and all. Not an
official statistic of the Instituto Nacional de Estadística.

## What it found

**The informality rate is 59.8%, not 63.8%.** Ignoring the expansion factors
moves the answer by **four percentage points** — not because the arithmetic is
hard, but because an unweighted mean describes the sample instead of the
country.

**A textbook confidence interval is 40% too narrow.** Households are surveyed
in clusters, so neighbours resemble each other and each extra respondent
carries less new information than an independent draw would. Resampling whole
clusters gives **58.3–61.4%**; the binomial interval claims **62.9–64.7%** —
narrower, and centred in the wrong place.

| | Naive | Design-based |
| --- | --- | --- |
| 95% interval | 62.9–64.7% | **58.3–61.4%** |
| Width | 1.83 pts | **3.05 pts** (1.67×) |
| Effective sample | 10,604 | **3,808** |

Clustering costs about **64% of the apparent information**. The nominal sample
of 10,604 carries what a simple random sample of 3,808 would.

**Firm size explains almost everything else.** A gradient-boosted model reaches
**0.860 AUC**. Removing firm size costs **0.094 AUC**; removing anything else
costs at most 0.009. Informality in Paraguay's private sector is overwhelmingly
a property of *where you work*, not of who you are:

| Feature removed | AUC lost |
| --- | --- |
| **Firm size** | **+0.0940** |
| Industry | +0.0085 |
| Years of schooling | +0.0070 |
| Age / Department | +0.0026 |
| Language at home | +0.0012 |
| Sex / Weekly hours | +0.0003 |
| Urban or rural | −0.0002 |

Workers in firms of 2–5 people are **91.9%** informal; in firms above 500,
**12.5%**. Urban/rural contributes nothing once firm size is known — the
apparent geography of informality is the geography of small firms.

**The clustering did not inflate the model score.** This was the hypothesis the
project set out to test, and it came back negative: splitting folds randomly
rather than by cluster changes AUC by **+0.0007** (logistic) and **+0.0018**
(boosting). Measured, reported, and kept here rather than quietly dropped. The
reason appears to be that the features are individual and job characteristics
rather than neighbourhood effects, so having seen a respondent's neighbours
adds nothing the features do not already say.

That the check came back clean is not a reason to skip it. The design effect on
the *interval* is real and large; the leakage effect on the *model* is not. Only
running both says which.

## Two coding traps in the raw data

Both produce plausible, wrong numbers with no error.

**Binary answers are coded 1/6**, not 1/0 or 1/2. Sex is `1` = male, **`6`** =
female; area is `1` = urban, **`6`** = rural. Treated as numbers, "female"
becomes six times "male". Every such variable is handled as a category here.

**Daily hours are `H.MM`, not decimal hours.** Eight and a half hours is
recorded as `8.30`, three quarters as `.45` — the variable label says "Horas y
Minutos". Adding the seven days arithmetically reads 8.30 as 8.3 and
understates the week. A test pins this: seven days of `8.30` is **59.5** hours,
and the naive sum says 58.1.

## Why private-sector employees

Informality across *all* employed workers is close to deterministic and
modelling it would measure nothing:

| Employment category | Informal |
| --- | --- |
| Self-employed | 99.9% |
| Unpaid family worker | 100.0% |
| Employer | 99.1% |
| Domestic worker | 94.6% |
| **Private-sector employee** | **59.8%** |
| Public-sector employee | 20.1% |

A classifier handed the employment category would score near-perfectly by
memorising those rows. Restricting to private-sector employees leaves a
population where the outcome is genuinely uncertain and the question has
content. The restriction is in `config/survey.yml` with this reasoning
attached, not buried in code.

## Definition

Informality is **not contributing to a pension fund** (`B10` in the survey),
which is the operational definition used by the ILO and by the INE itself.
Non-responses are dropped, never imputed: inventing a labour status is not a
modelling choice.

## Quick start

```bash
pip install -r requirements.txt
python -m informality build
```

Downloads the 2024 person file (16 MB SPSS, cached after the first run),
estimates the rate with a cluster bootstrap, fits both models under both
splitting schemes, runs 13 checks and writes
`data/reports/informality-report.html`.

```bash
python -m informality build --publish-docs   # also write docs/index.html
python -m informality build --refresh        # re-download instead of cache
python -m informality export                 # CSVs
pytest                                       # 16 tests, no network
```

The microdata is public and needs no registration, but the portal rejects a
bare client; the `User-Agent` is set in `config/survey.yml`.

## How it works

```
INE EPHC microdata (.sav, with its own value labels)
        │
   universe  ────► 57,744 → 10,604, each step recorded
        │
   features  ────► H.MM hours decoded, 1/6 codes kept categorical
        │
        ├──► weighted rate ──► cluster bootstrap ──► design effect
        │
        └──► 5-fold CV, weighted ──► random split    (the usual way)
                                 └─► clustered split (the honest way)
                                       │
                               leave-one-feature-out contributions
```

SPSS files are read rather than the parallel CSVs because they carry the
codebook inside them — the labels cannot drift out of step with the data.

## Checks

13 checks in three families.

| Family | What a failure means | Example |
| --- | --- | --- |
| **Sample** | the rows are not what the survey says | the expansion factors must reproduce Paraguay's 6.2m population to within 5% |
| **Design** | the sampling design was mishandled | a design interval *narrower* than the naive one would mean the bootstrap is resampling rows, not clusters |
| **Model** | the model is not what it claims | it must beat predicting the base rate for everyone |

The weights reproduce **6,126,659** people against the INE's ~6.2m projection.
That single number is the strongest evidence that the weights and the filters
are both right.

12 of 13 pass. The one finding is informational: weighted and unweighted rates
differ by 4.07 points — which is the error this project exists to avoid, so it
is recorded rather than hidden.

## Limitations

* **One year.** 2024 only. The config lists 2022 and 2023 and the loader takes
  a year argument, but no trend analysis is done and none should be read in.
* **No explicit strata.** The INE does not publish a stratum variable with the
  microdata. The bootstrap resamples clusters without stratifying, which is
  slightly conservative — the intervals here are, if anything, a little wide.
* **Pension contribution is a proxy.** It is the standard one, but it misses
  workers who are formal by contract and not enrolled, and those enrolled
  through a scheme the question does not capture.
* **Cross-sectional.** Nothing here is causal. Firm size predicts informality;
  this says nothing about what would happen if firms grew.
* **The 10,604 are a subset of a subset.** Self-employment is where most
  Paraguayan informality actually lives, and it is excluded here precisely
  because it is not predictable. The headline rate is **not** the national
  informality rate.
* **Weekly hours are self-reported** and two respondents report over 112 hours.
  Flagged, not clipped.
* **AUC is not accuracy.** The model ranks workers by risk; it is not a
  decision rule, and no threshold is proposed.

## Layout

```
config/survey.yml            source, design, universe, features — with reasoning
src/informality/
  data.py        download, cache, universe filtering with an audit trail
  features.py    H.MM hour decoding, 1/6 coding handled as categories
  survey.py      weighted estimates, cluster bootstrap, design effect
  modelling.py   both CV schemes, leave-one-feature-out contributions
  checks.py      sample, design and model checks
  report.py      self-contained HTML
  pipeline.py    the whole run, as one object
tests/           16 tests, no network
docs/index.html  the generated report, served by GitHub Pages
```

## Tests

```bash
pytest
```

Three carry the argument:

* `test_clustering_widens_the_interval` — on a synthetic sample with known
  within-cluster similarity, the design interval must come out materially
  wider.
* `test_without_clustering_the_two_intervals_agree` — the control. Without it,
  the test above would prove only that the code always widens intervals.
* `test_the_naive_reading_understates_the_week` — pins the `H.MM` trap at
  59.5 hours against the naive 58.1.

---

Built by Sergio Alegre. Microdata from the
[Instituto Nacional de Estadística](https://www.ine.gov.py/microdatos/),
Paraguay.
