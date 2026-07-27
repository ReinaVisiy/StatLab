# StatLab — Statistical Analysis Suite

StatLab is a Python/Streamlit application for performing classical
statistical calculations — probability laws, descriptive statistics,
hypothesis tests, ANOVA, correlation, regression, and goodness-of-fit
testing — with full step-by-step working shown for every result, not
just a final number.

Every statistical procedure is implemented as a pure, self-contained
function (validate inputs → compute → return a structured result),
built on `scipy.stats` distribution objects and `statsmodels`, so no
statistical table or formula is ever hardcoded.

> **Status:** complete. The calculation engine (probability laws,
> descriptive statistics, every hypothesis test, ANOVA, correlation,
> regression, and goodness-of-fit), the full Streamlit front-end
> (`app.py`, 3-level navigation, all data-entry and results pages),
> full English/French localization (every computed step, hypothesis,
> conclusion, and validation error — not just static UI labels), and
> CSV/PNG/PDF export are all built, wired together, and verified to
> run end-to-end with no errors.

## Features

- **Discrete probability laws** — Bernoulli, Binomial, Poisson,
  Geometric, Negative Binomial, Hypergeometric, Discrete Uniform,
  Multinomial. PMF, CDF, survival, range, and inverse queries with
  full properties (mean, variance, mode, median, skewness) and
  PMF plots.
- **Continuous probability laws** — Normal, Standard Normal,
  Student's t, Chi-Square, F, Exponential, Continuous Uniform, Gamma,
  Beta, Lognormal, Cauchy, Laplace. PDF, CDF, survival, range, and
  inverse (ppf) queries with full properties and PDF plots.
- **Descriptive statistics** — discrete and continuous univariate
  statistics (mean, mode, median, variance, quartiles, coefficient of
  variation, skewness, kurtosis, Gini index, Lorenz curve), automatic
  class continuity correction, mass-frequency and mass-median
  calculation for grouped continuous data, and bivariate statistics
  (covariance, correlation, inter/intra-variance).
- **Parametric tests** — one/two-sample and paired z/t-tests,
  one/two-proportion z-tests, F-test for variance equality,
  confidence intervals, sample size calculator. Every test supports
  both raw-data and summary-statistics input.
- **Non-parametric tests** — Mann-Whitney U, Wilcoxon signed-rank,
  Kruskal-Wallis, Sign test, Runs test.
- **ANOVA** — one-way (with Tukey HSD post-hoc), two-way without
  replication, two-way with replication (interaction tested first),
  Bartlett's and Levene's tests for variance homogeneity.
- **Correlation** — Pearson, Kendall's tau, Spearman, and a full
  correlation matrix with eigen-decomposition.
- **Regression** — simple linear, multiple linear, and polynomial
  regression, each with full ANOVA breakdown, residual diagnostics,
  and prediction.
- **Goodness of fit** — chi-square GOF test for every discrete and
  continuous law above, plus the chi-square test of independence
  (with Cramér's V effect size).
- **Bilingual UI text** — English/French display strings throughout,
  including hypotheses, conclusions, assumption checks, and validation
  error messages (code and logic remain in English).
- **Exports** — CSV download for every data/result table, PNG download
  for every plot, and a full PDF report per result (hypotheses,
  decision, statistic/p-value/critical-value, conclusion, properties,
  the plot, full step-by-step working, and any extra tables).

## Statistical methods implemented

| Category | Methods |
|---|---|
| Discrete laws | Bernoulli, Binomial, Poisson, Geometric, Negative Binomial, Hypergeometric, Discrete Uniform, Multinomial |
| Continuous laws | Normal, Standard Normal, Student's t, Chi-Square, F, Exponential, Continuous Uniform, Gamma, Beta, Lognormal, Cauchy, Laplace |
| Descriptive | Discrete/continuous univariate stats, bivariate stats, Gini/Lorenz |
| Parametric tests | z-test (1/2-sample, 1/2-proportion), t-test (1/2-sample, paired), F-test, confidence intervals, sample size |
| Non-parametric | Mann-Whitney, Wilcoxon signed-rank, Kruskal-Wallis, Sign test, Runs test |
| ANOVA | One-way (+Tukey HSD), two-way (no replication / with replication), Bartlett, Levene |
| Correlation | Pearson, Kendall's tau, Spearman, correlation matrix |
| Regression | Simple linear, multiple linear, polynomial |
| Goodness of fit | Chi-square GOF for all laws above, chi-square independence test |

## Install & run locally

```bash
git clone https://github.com/ReinaVisiy/StatLab.git
cd StatLab
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Running tests

A pytest suite in `test_suite/` checks known-answer inputs (and, where
applicable, cross-checks against scipy directly) for the core discrete/
continuous laws, descriptive stats, parametric/non-parametric tests, and
ANOVA, plus a registry smoke test that every registered tool actually
imports and runs. It doesn't yet cover correlation, regression, or the
goodness-of-fit modules — contributions welcome.

```bash
pip install -r requirements.txt pytest
pytest test_suite/ -q
```

## Usage guide

1. From the home page, pick a suite (e.g. *Parametric Tests*).
2. Pick a specific law or test from that suite's list.
3. Enter your data or parameters (raw data, summary statistics, or
   distribution parameters, depending on the tool) and choose your
   significance level / tail direction where relevant.
4. Press **Calculate** to see the full results page: hypotheses,
   assumptions, step-by-step working, test statistic, critical value,
   p-value, decision, and plot.
5. Use **Back to edit** to return to your inputs (they're preserved)
   and adjust a value without starting over.
6. Download any table as CSV, any plot as PNG, or the whole result as
   a PDF report, directly from the results page.

No sample or preloaded data is bundled — all input comes from what
you enter or upload.

## File structure

```
StatLab/
├── app.py                          # Main entry point
├── core/
│   ├── helpers.py                  # Shared utilities, plotting, formatting, downloads
│   ├── param_validation.py         # Input validation helpers
│   ├── registry.py                 # Suite/tool registry powering the 3-level nav
│   ├── ui_engine.py                # Streamlit page rendering (entry/results pages)
│   └── report_pdf.py               # Full-report PDF export
├── i18n/
│   └── translations.py             # EN/FR display strings
├── laws/
│   ├── discrete/                   # 8 discrete probability laws
│   └── continuous/                 # 12 continuous probability laws
├── descriptive/                    # Univariate/bivariate descriptive stats
├── tests/
│   ├── parametric/                 # z/t/F tests, CI, sample size
│   ├── nonparametric/              # Mann-Whitney, Wilcoxon, Kruskal-Wallis...
│   ├── anova/                      # One-way, two-way, Bartlett, Levene
│   ├── correlation/                # Pearson, Kendall, Spearman, matrix
│   ├── regression/                 # Simple/multiple/polynomial regression
│   └── goodness_of_fit/            # GOF test for every law, plus chi-square
│                                    # independence/homogeneity tests
├── test_suite/                     # pytest suite (148 tests)
├── requirements.txt
├── .gitignore
└── README.md
```

## Deployment (Streamlit Community Cloud)
Deployed!!!

url: https://statistics-laboratory.streamlit.app/

## License

MIT License

Copyright (c) 2026 Reina Visiy

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
