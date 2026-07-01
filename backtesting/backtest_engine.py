"""
backtest_engine.py

Simulates the trading strategy on OUT-OF-SAMPLE historical data only.
Includes realistic Indian market transaction costs and a market
trend filter — only trades in the direction of the Nifty 50 trend.

Costs included per trade:
  - Upstox brokerage : ₹20 flat per order (buy + sell = ₹40 per trade)
  - STT              : 0.025% of sell-side turnover
  - Exchange charges : ~0.005% both sides
  - Slippage         : 0.05% assumed on entry and exit
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# ── Indian market transaction costs ───────────────────────────────────────
BROKERAGE_PER_ORDER = 20
STT_RATE            = 0.00025
EXCHANGE_CHARGE     = 0.00005
SLIPPAGE_RATE       = 0.0005


def compute_transaction_cost(trade_value: float) -> float:
    """Total cost of one complete round-trip trade (buy + sell)."""
    brokerage = BROKERAGE_PER_ORDER * 2
    stt       = trade_value * STT_RATE
    exchange  = trade_value * EXCHANGE_CHARGE * 2
    slippage  = trade_value * SLIPPAGE_RATE * 2
    return brokerage + stt + exchange + slippage


class BacktestEngine:
    def __init__(
        self,
        initial_capital  : float = 100000,
        max_slots        : int   = 5,
        max_risk_pct     : float = 0.10,
        min_confidence   : float = 0.62,
        stop_loss_pct    : float = 0.015,
        take_profit_pct  : float = 0.045,
        trend_filter     : bool  = True,   # only trade with market trend
        trend_threshold  : float = 0.0,    # nifty_trend must exceed this
    ):
        self.initial_capital  = initial_capital
        self.capital          = initial_capital
        self.max_slots        = max_slots
        self.max_risk_pct     = max_risk_pct
        self.min_confidence   = min_confidence
        self.stop_loss_pct    = stop_loss_pct
        self.take_profit_pct  = take_profit_pct
        self.trend_filter     = trend_filter
        self.trend_threshold  = trend_threshold

        self.open_trades   = {}
        self.closed_trades = []
        self.equity_curve  = []

        # Tracking stats
        self.filtered_by_trend = 0

    def position_size(self) -> float:
        """Capital per trade — slot size capped at max_risk_pct."""
        slot_size = self.capital / self.max_slots
        max_size  = self.initial_capital * self.max_risk_pct
        return min(slot_size, max_size)

    def is_trend_aligned(self, signal: str, nifty_trend: float) -> bool:
        """
        Checks if the trade signal aligns with the market trend.

        BUY signals only taken when Nifty is in uptrend (nifty_trend > threshold)
        SELL signals only taken when Nifty is in downtrend (nifty_trend < -threshold)

        This prevents fighting the market — one of the most common reasons
        retail trading strategies lose money.
        """
        if not self.trend_filter:
            return True
        if signal == "BUY"  and nifty_trend >  self.trend_threshold:
            return True
        if signal == "SELL" and nifty_trend < -self.trend_threshold:
            return True
        return False

    def open_trade(self, symbol, signal, price, confidence, timestamp):
        """Opens a new trade position."""
        if symbol in self.open_trades:
            return
        if len(self.open_trades) >= self.max_slots:
            return

        size   = self.position_size()
        cost   = compute_transaction_cost(size)
        shares = size / price

        self.open_trades[symbol] = {
            "symbol"      : symbol,
            "signal"      : signal,
            "entry_price" : price,
            "shares"      : shares,
            "size"        : size,
            "cost"        : cost,
            "confidence"  : confidence,
            "entry_time"  : timestamp,
            "stop_loss"   : price * (1 - self.stop_loss_pct)  if signal == "BUY"
                            else price * (1 + self.stop_loss_pct),
            "take_profit" : price * (1 + self.take_profit_pct) if signal == "BUY"
                            else price * (1 - self.take_profit_pct),
        }

    def close_trade(self, symbol, exit_price, timestamp, reason="signal"):
        """Closes an open trade and records the result."""
        if symbol not in self.open_trades:
            return

        trade  = self.open_trades.pop(symbol)
        signal = trade["signal"]

        if signal == "BUY":
            gross_pnl = (exit_price - trade["entry_price"]) * trade["shares"]
        else:
            gross_pnl = (trade["entry_price"] - exit_price) * trade["shares"]

        net_pnl      = gross_pnl - trade["cost"]
        self.capital += net_pnl

        self.closed_trades.append({
            **trade,
            "exit_price" : exit_price,
            "exit_time"  : timestamp,
            "gross_pnl"  : round(gross_pnl, 2),
            "net_pnl"    : round(net_pnl, 2),
            "cost"       : round(trade["cost"], 2),
            "exit_reason": reason,
            "return_pct" : round(net_pnl / trade["size"] * 100, 3),
        })

    def check_stops(self, symbol, current_price, timestamp):
        """Checks if stop-loss or take-profit has been triggered."""
        if symbol not in self.open_trades:
            return

        trade  = self.open_trades[symbol]
        signal = trade["signal"]

        if signal == "BUY":
            if current_price <= trade["stop_loss"]:
                self.close_trade(symbol, current_price, timestamp, "stop_loss")
            elif current_price >= trade["take_profit"]:
                self.close_trade(symbol, current_price, timestamp, "take_profit")
        else:
            if current_price >= trade["stop_loss"]:
                self.close_trade(symbol, current_price, timestamp, "stop_loss")
            elif current_price <= trade["take_profit"]:
                self.close_trade(symbol, current_price, timestamp, "take_profit")

    def run(self, df: pd.DataFrame) -> dict:
        """
        Runs the backtest chronologically.
        Applies trend filter before opening any new position.
        """
        df = df.sort_values("timestamp").reset_index(drop=True)

        for _, row in df.iterrows():
            symbol      = row["symbol"]
            price       = row["close"]
            signal      = row["signal"]
            confidence  = row["confidence"]
            timestamp   = row["timestamp"]
            nifty_trend = row.get("nifty_trend", 0.0)

            # 1. Check stops on open positions first
            self.check_stops(symbol, price, timestamp)

            # 2. Open new trade if:
            #    - no existing position for this symbol
            #    - signal confidence is high enough
            #    - signal is aligned with the market trend
            if (symbol not in self.open_trades and
                    signal in ("BUY", "SELL") and
                    confidence >= self.min_confidence):

                if self.is_trend_aligned(signal, nifty_trend):
                    self.open_trade(symbol, signal, price, confidence, timestamp)
                else:
                    self.filtered_by_trend += 1

            # 3. Exit on signal reversal
            elif symbol in self.open_trades:
                open_signal = self.open_trades[symbol]["signal"]
                if ((open_signal == "BUY"  and signal == "SELL") or
                        (open_signal == "SELL" and signal == "BUY")):
                    self.close_trade(symbol, price, timestamp, "signal_reversal")

            # 4. Record equity at each step
            self.equity_curve.append({
                "timestamp": timestamp,
                "capital"  : round(self.capital, 2),
            })

        # Close remaining open trades at last price
        for symbol in list(self.open_trades.keys()):
            last_price = df[df["symbol"] == symbol]["close"].iloc[-1]
            self.close_trade(symbol, last_price,
                             df["timestamp"].iloc[-1], "end_of_data")

        return self.summary()

    def summary(self) -> dict:
        """Computes all performance metrics from closed trades."""
        trades = pd.DataFrame(self.closed_trades)

        if trades.empty:
            print("No trades were executed.")
            return {}

        total_trades = len(trades)
        winning      = trades[trades["net_pnl"] > 0]
        losing       = trades[trades["net_pnl"] < 0]
        win_rate     = len(winning) / total_trades * 100
        total_pnl    = trades["net_pnl"].sum()
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100

        # Sharpe ratio (annualized)
        daily_returns = trades["return_pct"] / 100
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                  if daily_returns.std() > 0 else 0)

        # Max drawdown
        equity       = pd.Series([e["capital"] for e in self.equity_curve])
        rolling_max  = equity.cummax()
        drawdown     = (equity - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # Profit factor = gross wins / gross losses
        gross_wins   = winning["gross_pnl"].sum() if len(winning) > 0 else 0
        gross_losses = abs(losing["gross_pnl"].sum()) if len(losing) > 0 else 1
        profit_factor = round(gross_wins / gross_losses, 3)

        return {
            "initial_capital"    : self.initial_capital,
            "final_capital"      : round(self.capital, 2),
            "total_return_pct"   : round(total_return, 2),
            "total_pnl"          : round(total_pnl, 2),
            "total_trades"       : total_trades,
            "filtered_by_trend"  : self.filtered_by_trend,
            "win_rate_pct"       : round(win_rate, 2),
            "winning_trades"     : len(winning),
            "losing_trades"      : len(losing),
            "avg_win"            : round(winning["net_pnl"].mean(), 2) if len(winning) > 0 else 0,
            "avg_loss"           : round(losing["net_pnl"].mean(), 2)  if len(losing)  > 0 else 0,
            "profit_factor"      : profit_factor,
            "sharpe_ratio"       : round(sharpe, 3),
            "max_drawdown_pct"   : round(max_drawdown, 2),
            "total_costs_paid"   : round(trades["cost"].sum(), 2),
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.predict import load_model, predict_dataframe

    # Load TEST SET ONLY — data model never saw during training
    df = pd.read_csv(DATA_DIR / "daily_test.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(df)} out-of-sample rows for backtesting")
    print(f"Period: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}\n")

    # Generate signals
    print("Generating signals...")
    model, le = load_model()
    df_signals = predict_dataframe(model, le, df)

    print(f"Signal distribution:\n{df_signals['signal'].value_counts()}\n")

    # Run backtest with trend filter enabled
    print("Running backtest (with market trend filter)...")
    engine = BacktestEngine(
        initial_capital = 100000,
        max_slots       = 5,
        min_confidence  = 0.62,
        stop_loss_pct   = 0.015,
        take_profit_pct = 0.045,
        trend_filter    = True,
        trend_threshold = 0.0,
    )
    results = engine.run(df_signals)

    # Print results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS (OUT-OF-SAMPLE + TREND FILTER)")
    print("=" * 50)
    for key, val in results.items():
        print(f"  {key:<25} : {val}")

    # Save trade log
    trades_df = pd.DataFrame(engine.closed_trades)
    out_path  = DATA_DIR / "backtest_trades.csv"
    trades_df.to_csv(out_path, index=False)
    print(f"\nTrade log saved → {out_path}")