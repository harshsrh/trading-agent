# Model Status

## Current Performance (as of July 2026)

| Metric | Value |
|---|---|
| Out-of-sample signal accuracy | 45.08% |
| Backtest return (1 year, OOS) | -2.79% |
| Total trades | 19 |
| Win rate | 31.58% |

## Assessment

The current model does not have sufficient predictive edge for live trading.
It is deployed for **paper trading only** to validate the execution infrastructure.

## Known limitations

- Features limited to standard technical indicators (low uniqueness)
- No sector/macro context
- No earnings/event calendar
- Trained on only 5 stocks

## Planned improvements

- [ ] Add sector ETF relative strength features
- [ ] Add macroeconomic indicators (FII/DII flows, VIX India)
- [ ] Expand watchlist to 20+ stocks
- [ ] Experiment with LSTM for sequence modeling
- [ ] Test mean-reversion approach alongside trend-following