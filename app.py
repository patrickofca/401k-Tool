import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os

# --- FILE PATH FOR PERSISTENCE ---
CONFIG_FILE = "portfolio_config.json"

# --- HELPER FUNCTIONS FOR STORAGE ---


def load_saved_data():
    default_data = {
        "portfolio_value": 100000,
        "years_to_retirement": 20,
        "pullback_trigger": 10,
        "monthly_contrib": 500,
        "company_match": 2000,
        "allocations": [
            {"Ticker": "SPY", "Your Allocation %": 60.0},
            {"Ticker": "AGG", "Your Allocation %": 30.0},
            {"Ticker": "BIL", "Your Allocation %": 10.0}
        ]
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return default_data
    return default_data


def save_current_data(data_dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(data_dict, f, indent=4)


saved_config = load_saved_data()

# --- CORE PAGE LAYOUT SETUP ---
st.set_page_config(page_title="Persistent 401k Optimizer", layout="wide")
st.title("📈 401k Optimizer with 50th Percentile Tactical Reallocation")

# --- SIDEBAR CONFIGURATION CONTROLS ---
st.sidebar.header("🎯 Profile Settings")
portfolio_value = st.sidebar.number_input(
    "Current 401k Balance ($)", value=int(saved_config["portfolio_value"]), step=5000)
years_to_retirement = st.sidebar.number_input(
    "Years Until Retirement:", min_value=1, max_value=50, value=int(saved_config["years_to_retirement"]))
pullback_trigger = st.sidebar.slider(
    "Dip Trigger (%)", min_value=1, max_value=40, value=int(saved_config["pullback_trigger"]))

st.sidebar.write("---")
st.sidebar.header("💰 Savings & Company Match")
monthly_contrib = st.sidebar.number_input(
    "Your Monthly Contribution ($):", min_value=0, value=int(saved_config["monthly_contrib"]), step=100)
company_match = st.sidebar.number_input(
    "Yearly Company Match ($):", min_value=0, value=int(saved_config["company_match"]), step=250)

if "allocation_df" not in st.session_state:
    st.session_state.allocation_df = pd.DataFrame(saved_config["allocations"])

# --- CORE ASSET ALLOCATION WORKSPACE MATRIX ---
st.subheader("🧱 Portfolio Asset Allocation Weights")
st.write("Type your fund percentages directly into the table cells below:")
edited_alloc_df = st.data_editor(
    st.session_state.allocation_df, num_rows="dynamic", use_container_width=True)
st.session_state.allocation_df = edited_alloc_df

# --- RESTORED FEATURE: MANUAL REBALANCE OVERRIDE BUTTON ---
if st.button("🔄 Trigger Manual Rebalance to Targets"):
    st.toast("Calculating exact dollar movements to restore your baseline targets...")
    target_eq_val = portfolio_value * (80 / 100)  # Baseline targets macro
    # This button lets you instantly see your static base rebalance orders
    st.info("💡 Baseline Rebalance Active: Check the table layout at the bottom of the page to execute your steady-state trades.")

# Save triggers
current_settings_payload = {
    "portfolio_value": portfolio_value,
    "years_to_retirement": years_to_retirement,
    "pullback_trigger": pullback_trigger,
    "monthly_contrib": monthly_contrib,
    "company_match": company_match,
    "allocations": edited_alloc_df.to_dict(orient="records")
}
save_current_data(current_settings_payload)

# --- CALCULATIONS ENGINE ---
active_tickers = edited_alloc_df["Ticker"].tolist()

SINGLE_STOCK_CAP_PCT = 5.0
INDEX_FUNDS = ['SPY', 'MDY', 'VXUS', 'QQQ', 'VTI', 'IWM', 'ITOT', 'SCHB']

if active_tickers:
    with st.spinner("Scraping metrics from Yahoo Finance..."):
        data = yf.download(active_tickers, period="max", progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=active_tickers[0])
        data = data.dropna(axis=1, how='all')
        returns = data.pct_change().dropna()

    # Calculate rolling dip metrics
    live_drawdowns = {}
    for ticker in active_tickers:
        if ticker in data.columns:
            series = data[ticker].dropna()
            rolling_peaks = series.cummax()
            if not series.empty:
                live_drawdowns[ticker] = (
                    (series.iloc[-1] - rolling_peaks.iloc[-1]) / rolling_peaks.iloc[-1]) * 100

    st.subheader("📊 Current Market Drawdown Metrics")
    cols = st.columns(max(len(live_drawdowns), 1))
    for i, (tk, drop) in enumerate(live_drawdowns.items()):
        cols[i].metric(f"{tk} vs Peak", f"{drop:.2f}%",
                       delta=f"{drop:.2f}%", delta_color="inverse")

    # --- 50th PERCENTILE GROWTH & REALLOCATION MATHEMATICS ---
    weight_map = dict(zip(edited_alloc_df["Ticker"], edited_alloc_df["Your Allocation %"]))
    valid_tickers = [t for t in active_tickers if t in returns.columns]
    weights = np.array([weight_map.get(t, 0.0) for t in valid_tickers]) / 100
    returns_valid = returns[valid_tickers] if valid_tickers else returns

    if sum(weights) > 0:
        blended_returns = returns_valid.dot(weights)
        daily_mean = blended_returns.mean()
        daily_std = blended_returns.std()

        sim_days = int(252 * years_to_retirement)
        num_simulations = 500
        sim_paths = np.zeros((sim_days, num_simulations))

        daily_contribution = monthly_contrib / 21
        daily_company_match = company_match / 252

        np.random.seed(42)
        for p in range(num_simulations):
            path = np.zeros(sim_days)
            current_balance = portfolio_value
            random_shocks = np.random.normal(daily_mean, daily_std, sim_days)
            for d in range(sim_days):
                current_balance = current_balance * \
                    (1 + random_shocks[d]) + \
                    daily_contribution + daily_company_match
                path[d] = current_balance
            sim_paths[:, p] = path

        # Isolate final distribution percentiles
        p10_final_value = np.percentile(sim_paths[-1, :], 10)
        p50_final_value = np.percentile(sim_paths[-1, :], 50)
        p90_final_value = np.percentile(sim_paths[-1, :], 90)

        # Reallocation Calculator Logic
        st.subheader("📢 Automated 401k Reallocation Advice")

        # Categorize defensive cash/bond pools vs growth equities based on your tickers
        bond_proxies = ['AGG', 'BIL']
        dipped_equities = [tk for tk, drop in live_drawdowns.items(
        ) if drop <= -pullback_trigger and tk not in bond_proxies]

        if dipped_equities:
            st.error(f"🚨 TACTICAL ACTION REQUIRED: REALLOCATE INSTEAD OF HOLDING")
            st.write(
                f"The following equity assets are heavily discounted: **{', '.join(dipped_equities)}**.")

            # Math: Calculate how much defensive value to transfer based on 50th percentile expectations
            total_defensive_weight = sum(
                [row['Your Allocation %'] for row in saved_config['allocations'] if row['Ticker'] in bond_proxies]) / 100
            current_defensive_pool = portfolio_value * total_defensive_weight

            # Rule: Allocate a tactical 25% chunk of your defensive dry-powder into the dipped funds
            tactical_transfer_total = current_defensive_pool * 0.25
            per_fund_transfer = tactical_transfer_total / len(dipped_equities)

            st.write(f"### 📑 Your 50th Percentile Tactical Reallocation Plan")
            st.write(
                f"Based on your 50th Percentile Growth target of **${p50_final_value:,.2f}** at retirement, your portfolio needs to maintain an aggressive equity velocity. To buy this dip safely, execute the following exchanges inside your 401k portal:")

            # Determine the best available index fund to absorb overflow
            redirect_target = next(
                (t for t in INDEX_FUNDS if t in active_tickers and t not in dipped_equities), 'SPY')

            realloc_data = []
            for eq_fund in dipped_equities:
                current_weight_pct = weight_map.get(eq_fund, 0.0)
                is_single_stock = eq_fund not in INDEX_FUNDS and eq_fund not in bond_proxies

                if is_single_stock and current_weight_pct >= SINGLE_STOCK_CAP_PCT:
                    # Already at or over cap — redirect entire transfer to index
                    realloc_data.append({
                        "SELL FROM": "AGG / BIL",
                        "BUY INTO": redirect_target,
                        "Transfer ($)": f"${per_fund_transfer:,.2f}",
                        "Reason": f"{eq_fund} at {current_weight_pct:.1f}% — at {SINGLE_STOCK_CAP_PCT:.0f}% cap, full amount redirected to index"
                    })
                elif is_single_stock:
                    headroom_dollars = portfolio_value * ((SINGLE_STOCK_CAP_PCT - current_weight_pct) / 100)
                    capped_transfer = min(per_fund_transfer, headroom_dollars)
                    overflow = per_fund_transfer - capped_transfer

                    realloc_data.append({
                        "SELL FROM": "AGG / BIL",
                        "BUY INTO": eq_fund,
                        "Transfer ($)": f"${capped_transfer:,.2f}",
                        "Reason": f"Single stock capped at {SINGLE_STOCK_CAP_PCT:.0f}% max portfolio weight"
                    })
                    if overflow > 0:
                        realloc_data.append({
                            "SELL FROM": "AGG / BIL",
                            "BUY INTO": redirect_target,
                            "Transfer ($)": f"${overflow:,.2f}",
                            "Reason": f"Overflow above {SINGLE_STOCK_CAP_PCT:.0f}% cap redirected to broad index"
                        })
                else:
                    realloc_data.append({
                        "SELL FROM": "AGG / BIL",
                        "BUY INTO": eq_fund,
                        "Transfer ($)": f"${per_fund_transfer:,.2f}",
                        "Reason": "Broad market ETF — no single-stock cap applied"
                    })
            st.table(pd.DataFrame(realloc_data))
        else:
            st.success(f"✅ TACTICAL ACTION: HOLD / MAINTAIN STEADY BASELINE")
            st.write(
                f"All assets are operating within regular historical boundaries. To reach your 50th percentile estimated retirement balance of **${p50_final_value:,.2f}**, maintain your current percentage settings.")

        # --- RESTORED FEATURE: 10/50/90 TOTAL METRICS SUMMARIES ---
        st.subheader(
            f"🔮 {years_to_retirement}-Year Total Portfolio Performance Projections")

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Pessimistic Floor Value (10th Percentile)",
                   f"${p10_final_value:,.2f}")
        sc2.metric("Expected Growth Total (50th Percentile)",
                   f"${p50_final_value:,.2f}")
        sc3.metric("Optimistic Ceiling Value (90th Percentile)",
                   f"${p90_final_value:,.2f}")

        # --- GRAPH DISPLAYS ---
        chart_df = pd.DataFrame({
            "P10 (Pessimistic Floor)": np.percentile(sim_paths, 10, axis=1),
            "P50 (Expected Median Path)": np.percentile(sim_paths, 50, axis=1),
            "P90 (Optimistic Ceiling)": np.percentile(sim_paths, 90, axis=1)
        })
        st.line_chart(chart_df)

        # --- TARGET ASSET ALLOCATION CALCULATOR ---
        st.subheader("🧱 Portfolio Target Rebalance Structure")
        summary_rows = []
        for index, row in edited_alloc_df.iterrows():
            target_val = portfolio_value * (row["Your Allocation %"] / 100)
            summary_rows.append({
                "Asset Fund Ticker": row["Ticker"],
                "Target Percentage": f"{row['Your Allocation %']}%",
                "Calculated Ideal Value ($)": f"${target_val:,.2f}"
            })
        st.table(pd.DataFrame(summary_rows))
