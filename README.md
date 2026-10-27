# 📊 Instagram Analytics — Engagement & Impression Analysis

A Python data-analysis project that turns a raw Instagram post export into
engagement insights, correlation analysis and a small ML benchmark — with a
standalone HTML dashboard for browsing the results.

## What it does

- **15+ engagement metrics** per post — reach, impressions, likes, comments,
  saves, shares, profile visits, follows and conversion rates
- **Reach-source breakdown** — home feed, hashtags, explore, other
- **Correlation analysis** across metrics to see what actually moves reach
- **Top vs. average post benchmarking** to identify repeatable formats
- **Caption text analysis** (word frequency / word cloud)
- **ML benchmark** — Linear Regression and Random Forest models predicting
  impressions from engagement features
- **Interactive dashboard** — a self-contained `dashboard.html`

## Included dataset

`Instagram data.csv` — a post-level export of **120+ posts** totalling
**629,640 impressions**.

The impression-prediction models are reported in `PROJECT_SUMMARY.md` at
**R² ≈ 0.87**. That is a benchmark on this one dataset, not a generalisation
claim — the sample is small and comes from a single account.

## Project layout

```text
.
├── instagram_analytics.py   # main analytics module
├── run_analysis.py          # CLI entry point
├── ins pr.py                # original pre-refactor script
├── Instagram data.csv       # dataset (120+ posts)
├── dashboard.html           # self-contained results dashboard
├── insights/                # generated analysis reports
├── PROJECT_SUMMARY.md       # detailed write-up of the analysis
├── CITATION.cff             # citation metadata
└── .gitignore
```

## Quickstart

The repo ships no `requirements.txt`, so install the direct dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install pandas numpy matplotlib seaborn plotly scikit-learn wordcloud
python run_analysis.py
```

Then open `dashboard.html` in a browser and read the generated reports under
`insights/`.

## Notes

- The dataset is the author's own post export — no scraping and no credentials
  are included.
- The ML models are intentionally simple and exist as a comparison baseline,
  not as a production predictor.

## Author

**Dhyey Teraiya** — [GitHub](https://github.com/DhyeyTeraiya)

## Citation

See [`CITATION.cff`](CITATION.cff).
