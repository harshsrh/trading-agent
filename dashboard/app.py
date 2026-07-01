"""
app.py

Local Streamlit dashboard for the intraday AI trading agent.
Shows live signals, model reasoning, positions, trade history, and equity curve.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from datetime import datetime

# ── path setup ─────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT))

from models.predict import load_model, predict_dataframe, FEATURE_COLUMNS
from data.feature_engineering import FEATURE_COLUMNS as FE_FEATURES

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AI Trading Agent",
    page_icon  = "📈",
    layout     = "wide",
)

# ── helper functions ────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    """Load model once and cache it — avoids reloading on every interaction."""
    return load_model()


def load_backtest_trades() -> pd.DataFrame:
    path = DATA_DIR / "backtest_trades.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    return pd.DataFrame()


def load_featured_data() -> pd.DataFrame:
    path = DATA_DIR / "daily_featured.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    return pd.DataFrame()


def load_test_data() -> pd.DataFrame:
    path = DATA_DIR / "daily_test.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    return pd.DataFrame()


def signal_color(signal: str) -> str:
    return {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal, "⚪")


def build_equity_curve(trades: pd.DataFrame, initial_capital: float = 100000) -> pd.DataFrame:
    """Rebuilds equity curve from trade history."""
    if trades.empty:
        return pd.DataFrame()

    trades = trades.sort_values("exit_time").copy()
    equity = initial_capital
    curve  = [{"time": trades["entry_time"].min(), "capital": equity}]

    for _, t in trades.iterrows():
        equity += t["net_pnl"]
        curve.append({"time": t["exit_time"], "capital": round(equity, 2)})

    return pd.DataFrame(curve)


def explain_signal(model, le, feature_row: pd.Series) -> dict:
    """
    Generates SHAP-based explanation for a single prediction.
    Returns top 3 contributing features and their directions.
    """
    try:
        import shap
        X = feature_row[FEATURE_COLUMNS].values.reshape(1, -1)

        # Use a small background sample for speed
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)

        # Get predicted class
        proba      = model.predict_proba(X)[0]
        class_idx  = np.argmax(proba)
        signal_int = le.inverse_transform([class_idx])[0]

        feature_shap = shap_vals[0, :, class_idx]

        impact = sorted(
            zip(FEATURE_COLUMNS, feature_shap),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        reasons = []
        for feat, val in impact[:3]:
            reasons.append({
                "feature"   : feat,
                "value"     : round(float(feature_row[feat]), 4),
                "shap"      : round(float(val), 4),
                "direction" : "↑ bullish" if val > 0 else "↓ bearish",
            })

        return {"signal_int": signal_int, "reasons": reasons}

    except Exception as e:
        return {"signal_int": 0, "reasons": [], "error": str(e)}


# ── main dashboard ──────────────────────────────────────────────────────────
def main():
    # Header
    st.title("📈 AI Trading Agent — Dashboard")
    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
               f"Mode: Paper Trading  |  Market: NSE")

    st.divider()

    # ── load data ───────────────────────────────────────────────────────────
    model, le     = get_model()
    trades        = load_backtest_trades()
    featured_data = load_featured_data()
    test_data     = load_test_data()

    # Generate signals on test data
    if not test_data.empty:
        df_signals = predict_dataframe(model, le, test_data)
    else:
        df_signals = pd.DataFrame()

    # ── section 1: portfolio overview ───────────────────────────────────────
    st.subheader("Portfolio Overview")

    initial_capital = 100000
    total_pnl       = trades["net_pnl"].sum() if not trades.empty else 0
    final_capital   = initial_capital + total_pnl
    total_return    = (total_pnl / initial_capital) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Initial Capital",
        f"₹{initial_capital:,.0f}",
    )
    col2.metric(
        "Current Capital",
        f"₹{final_capital:,.2f}",
        delta=f"₹{total_pnl:,.2f}",
    )
    col3.metric(
        "Total Return",
        f"{total_return:.2f}%",
        delta=f"{total_return:.2f}%",
    )
    col4.metric(
        "Total Trades",
        len(trades) if not trades.empty else 0,
    )

    st.divider()

    # ── section 2: live signals ─────────────────────────────────────────────
    st.subheader("Latest Model Signals")
    st.caption("Most recent signal per stock from the out-of-sample period")

    if not df_signals.empty:
        # Get the latest signal per symbol
        latest = (
            df_signals.sort_values("timestamp")
            .groupby("symbol")
            .last()
            .reset_index()
        )

        cols = st.columns(len(latest))
        for i, (_, row) in enumerate(latest.iterrows()):
            icon = signal_color(row["signal"])
            cols[i].metric(
                label = row["symbol"].replace(".NS", ""),
                value = f"{icon} {row['signal']}",
                delta = f"{row['confidence']:.1%} confidence",
            )

        st.divider()

        # ── section 3: model reasoning ──────────────────────────────────────
        st.subheader("Model Reasoning")
        st.caption("Select a stock to see why the model made its latest signal")

        selected = st.selectbox(
            "Choose stock:",
            options=latest["symbol"].tolist(),
            format_func=lambda x: x.replace(".NS", ""),
        )

        if selected:
            row         = latest[latest["symbol"] == selected].iloc[0]
            signal_icon = signal_color(row["signal"])

            st.markdown(f"### {signal_icon} {row['signal']} — "
                        f"{selected.replace('.NS','')} "
                        f"({row['confidence']:.1%} confidence)")

            with st.spinner("Computing SHAP explanation..."):
                explanation = explain_signal(model, le, row)

            if explanation.get("reasons"):
                st.markdown("**Top 3 reasons for this signal:**")

                for r in explanation["reasons"]:
                    bar_width = min(int(abs(r["shap"]) * 1000), 100)
                    color     = "#00cc44" if r["shap"] > 0 else "#ff4444"
                    st.markdown(
                        f"""
                        <div style='margin-bottom:12px;'>
                            <b>{r['feature']}</b> = {r['value']}
                            &nbsp;&nbsp;<span style='color:{color}'>{r['direction']}</span>
                            &nbsp;&nbsp;(impact: {r['shap']:+.4f})
                            <div style='background:{color};width:{bar_width}%;
                                        height:8px;border-radius:4px;margin-top:4px;'>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("SHAP explanation not available for this signal.")

            # Show full probability breakdown
            st.markdown("**Full signal probability breakdown:**")
            prob_cols = st.columns(3)
            proba = model.predict_proba(
                row[FEATURE_COLUMNS].values.reshape(1, -1)
            )[0]
            classes = le.inverse_transform([0, 1, 2])
            labels  = {-1: "SELL 🔴", 0: "HOLD 🟡", 1: "BUY 🟢"}

            for i, (cls, prob) in enumerate(zip(classes, proba)):
                prob_cols[i].metric(labels[cls], f"{prob:.1%}")

    else:
        st.info("No signal data available. Run the pipeline first.")

    st.divider()

    # ── section 4: equity curve ─────────────────────────────────────────────
    st.subheader("Equity Curve")

    if not trades.empty:
        equity_df = build_equity_curve(trades, initial_capital)

        if not equity_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x    = equity_df["time"],
                y    = equity_df["capital"],
                mode = "lines",
                name = "Portfolio Value",
                line = dict(color="#00cc44" if total_pnl >= 0 else "#ff4444", width=2),
                fill = "tozeroy",
                fillcolor = "rgba(0,204,68,0.1)" if total_pnl >= 0 else "rgba(255,68,68,0.1)",
            ))
            fig.add_hline(
                y          = initial_capital,
                line_dash  = "dash",
                line_color = "gray",
                annotation_text = "Initial Capital",
            )
            fig.update_layout(
                xaxis_title = "Date",
                yaxis_title = "Portfolio Value (₹)",
                height      = 350,
                margin      = dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, width="stretch")

    else:
        st.info("No trade history yet — run the backtester to populate this.")

    st.divider()

    # ── section 5: trade history ─────────────────────────────────────────────
    st.subheader("Trade History")

    if not trades.empty:
        # Performance summary row
        winning     = trades[trades["net_pnl"] > 0]
        losing      = trades[trades["net_pnl"] < 0]
        win_rate    = len(winning) / len(trades) * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Win Rate",    f"{win_rate:.1f}%")
        m2.metric("Avg Win",     f"₹{winning['net_pnl'].mean():.2f}" if len(winning) > 0 else "—")
        m3.metric("Avg Loss",    f"₹{losing['net_pnl'].mean():.2f}"  if len(losing)  > 0 else "—")
        m4.metric("Best Trade",  f"₹{trades['net_pnl'].max():.2f}")
        m5.metric("Worst Trade", f"₹{trades['net_pnl'].min():.2f}")

        # Trade table
        display_cols = [
            "symbol", "signal", "entry_price", "exit_price",
            "net_pnl", "return_pct", "exit_reason", "entry_time",
        ]
        available = [c for c in display_cols if c in trades.columns]
        display   = trades[available].copy()
        display   = display.sort_values("entry_time", ascending=False)

        # Color the net_pnl column
        def color_pnl(val):
            color = "color: #00cc44" if val > 0 else "color: #ff4444"
            return color

        st.dataframe(
            display.style.map(color_pnl, subset=["net_pnl"]),
            width="stretch",
            height=300,
        )

        # Exit reason breakdown
        st.markdown("**Exit reason breakdown:**")
        reason_counts = trades["exit_reason"].value_counts()
        fig2 = px.bar(
            x      = reason_counts.index,
            y      = reason_counts.values,
            labels = {"x": "Exit Reason", "y": "Count"},
            color  = reason_counts.values,
            color_continuous_scale = "RdYlGn",
        )
        fig2.update_layout(
            height          = 250,
            margin          = dict(l=0, r=0, t=10, b=0),
            showlegend      = False,
            coloraxis_showscale = False,
        )
        st.plotly_chart(fig2, width="stretch")

    else:
        st.info("No trades yet.")

    st.divider()

    # ── section 6: model status ─────────────────────────────────────────────
    st.subheader("Model Status")

    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.markdown("""
        | Metric | Value |
        |---|---|
        | Model type | XGBoost Classifier |
        | Features | 16 (12 technical + 4 market context) |
        | Training period | Aug 2021 → Jul 2025 |
        | OOS signal accuracy | 45.08% |
        | Mode | 🟡 Paper Trading |
        """)

    with status_col2:
        st.warning("""
        ⚠️ **Paper Trading Mode**

        This agent is running in paper trading mode only.
        The model accuracy (45%) is below the threshold
        required for live capital deployment (>52%).

        Live trading will only be enabled after:
        - Model accuracy > 52% on OOS data
        - Positive backtest return over 3+ months
        - Profit factor > 1.2
        """)

    # Auto-refresh button
    st.divider()
    if st.button("🔄 Refresh Dashboard"):
        st.cache_resource.clear()
        st.rerun()


if __name__ == "__main__":
    main()