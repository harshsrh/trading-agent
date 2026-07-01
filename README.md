# Intraday AI Trading Agent

A self-built, personal-use algorithmic trading agent for the Indian stock market (NSE).
The agent uses a single machine learning model to generate intraday trading signals,
with full transparency into its reasoning, risk-managed position sizing, and local
execution via broker APIs.

## Project Goals

- Single ML model for signal generation (no ensemble of strategies)
- Intraday holding periods (minutes to hours, not scalping)
- Capital split across multiple concurrent trades to manage risk
- Full transparency: every trade and the reasoning behind it is logged and visible
- Runs entirely locally — no cloud hosting, personal use only
- SEBI-compliant: self-built algo for personal/immediate-family use, under the
  retail algo trading framework effective April 2026

## Status

🚧 Early development — building step by step.

## Architecture

| Stage | Module | Purpose |
|---|---|---|
| 1 | `data/` | Fetch historical & live OHLCV data, engineer technical indicator features |
| 2 | `models/` | Train a single classifier, generate predictions, explain reasoning (SHAP) |
| 3 | `backtesting/` | Simulate the strategy on historical data with realistic fees/slippage |
| 4 | `risk_management/` | Position sizing, capital allocation, stop-loss rules |
| 5 | `execution/` | Connect to broker API for paper/live order execution |
| 6 | `dashboard/` | Local dashboard showing trades, P&L, and live reasoning |

## Setup

\```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then fill in your own broker API credentials
\```

## Disclaimer

This is a personal/educational project. Trading involves real financial risk.
Past backtest performance does not guarantee future results. Built and used
for personal, non-commercial purposes only.