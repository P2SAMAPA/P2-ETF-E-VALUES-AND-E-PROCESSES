# E-Values and E-Processes for ETFs

Implements the emerging alternative to p-values. E-values are nonnegative random variables with expectation ≤1 under the null. E-processes allow optional stopping and optional continuation – natural for sequential market monitoring without multiple testing penalties. The per‑ETF score is the e-value.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Likelihood ratio e-values
- E-processes with optional stopping
- Macro-adjusted e-values
- Score = e-value (higher = stronger evidence)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-e-values-processes-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High e-value → strong evidence for positive expected return.
- E-value > 1 → evidence for alternative hypothesis.
- E-value ≤ 1 → no evidence.

## Requirements

See `requirements.txt`.
