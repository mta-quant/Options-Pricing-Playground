"""
Black-Scholes Options Pricing & Analysis Platform

A comprehensive options analysis tool featuring:
- Black-Scholes pricing
- Greeks calculation and visualization
- Implied volatility solver
- Payoff diagrams
- Strategy builder
- Monte Carlo simulation
- Scenario analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Import custom modules
import black_scholes as bs
import greeks
import iv as iv_module
import payoffs
import strategies
import simulation
import ui_components as ui

# Page configuration
st.set_page_config(
    page_title="Options Pricing Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Black-Scholes Options Pricing & Analysis Platform")
st.markdown("""
A comprehensive options analysis tool with pricing, Greeks, implied volatility, strategies, and Monte Carlo simulation.
""")

# Sidebar inputs
params = ui.create_parameter_inputs()

# Show welcome message if parameters not set
if params is None:
    st.markdown("---")
    st.info("**Welcome!** Please enter the required parameters in the left sidebar to begin analysis.")

    st.markdown("### Getting Started")
    st.markdown("""
    To use this platform, please provide the following inputs in the sidebar:

    1. **Current Stock Price (S)** - The current market price of the underlying asset
    2. **Strike Price (K)** - The option strike price (use Quick Presets for ATM/OTM/ITM)
    3. **Time to Expiration (T)** - Years until option expiration or select a calendar date
    4. **Volatility (σ)** - Annualized volatility percentage
    5. **Risk-Free Rate (r)** - Annualized risk-free interest rate percentage

    Once all required parameters are entered, you can explore:
    - **Pricing & Greeks**: View option prices and risk metrics
    - **Heatmaps**: Visualize sensitivities across price and volatility ranges
    - **Implied Volatility**: Calculate implied volatility from market prices
    - **Payoff Diagrams**: Understand profit/loss profiles
    - **Strategy Builder**: Create and analyze multi-leg strategies
    - **Monte Carlo**: Validate prices using simulation
    - **Scenario Analysis**: Test sensitivity to parameter changes
    """)

    st.markdown("### Quick Example")
    st.markdown("""
    Try these sample inputs to get started:
    - Stock Price: $100
    - Strike Price: $100 (ATM)
    - Time to Expiration: 1 year
    - Volatility: 20%
    - Risk-Free Rate: 5%
    """)

    st.stop()

# Extract parameters
S, K, T, r, sigma = params['S'], params['K'], params['T'], params['r'], params['sigma']

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Pricing & Greeks",
    "Heatmaps",
    "Implied Volatility",
    "Payoff Diagrams",
    "Strategy Builder",
    "Monte Carlo",
    "Scenario Analysis"
])

# ============================================================================
# TAB 1: PRICING & GREEKS
# ============================================================================
with tab1:
    st.header("Option Pricing & Greeks")

    # Calculate prices
    call_price = bs.call_price(S, K, T, r, sigma)
    put_price = bs.put_price(S, K, T, r, sigma)

    # Display prices
    ui.display_option_prices(call_price, put_price)

    # Calculate Greeks
    greeks_call = greeks.calculate_all_greeks(S, K, T, r, sigma, 'call')
    greeks_put = greeks.calculate_all_greeks(S, K, T, r, sigma, 'put')

    st.markdown("---")

    # Display Greeks
    ui.display_greeks(greeks_call, greeks_put)

    # Add interpretations
    with st.expander("Understanding the Greeks"):
        st.markdown("""
        **Delta (Δ)**: Measures the rate of change of option price relative to underlying price.
        - Call delta: 0 to 1 (positive exposure)
        - Put delta: -1 to 0 (negative exposure)

        **Gamma (Γ)**: Measures the rate of change of delta. Higher gamma means delta changes faster.
        - Always positive for long options
        - Highest for at-the-money options

        **Vega (ν)**: Measures sensitivity to volatility changes (per 1% change).
        - Always positive for long options
        - Highest for at-the-money options with longer maturity

        **Theta (Θ)**: Measures time decay (per day).
        - Usually negative for long options (time decay hurts)
        - Accelerates as expiration approaches

        **Rho (ρ)**: Measures sensitivity to interest rate changes (per 1% change).
        - Positive for calls, negative for puts
        - More significant for longer-dated options
        """)

    # Download Greeks
    st.markdown("---")
    greeks_df = pd.DataFrame({
        'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
        'Call': [greeks_call[k] for k in ['delta', 'gamma', 'vega', 'theta', 'rho']],
        'Put': [greeks_put[k] for k in ['delta', 'gamma', 'vega', 'theta', 'rho']]
    })
    ui.download_button_csv(greeks_df, "greeks.csv", "Download Greeks as CSV")

# ============================================================================
# TAB 2: HEATMAPS
# ============================================================================
with tab2:
    st.header("Interactive Heatmaps")

    # Select metric and option type at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        metric = st.selectbox(
            "Select Metric",
            options=['Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
            help="Choose which metric to visualize"
        )
    with col2:
        option_type = st.radio("Option Type", ['Call', 'Put'], horizontal=True)
    with col3:
        grid_size = st.selectbox(
            "Grid Size",
            options=[10, 15, 20, 25, 30],
            index=0,
            help="Number of grid points"
        )

    # Initialize session state with default ranges
    if 'min_price_pct' not in st.session_state:
        st.session_state.min_price_pct = -20.0
    if 'max_price_pct' not in st.session_state:
        st.session_state.max_price_pct = 20.0
    if 'min_price_abs' not in st.session_state:
        st.session_state.min_price_abs = S * 0.8
    if 'max_price_abs' not in st.session_state:
        st.session_state.max_price_abs = S * 1.2
    if 'min_vol_pct' not in st.session_state:
        st.session_state.min_vol_pct = 10.0
    if 'max_vol_pct' not in st.session_state:
        st.session_state.max_vol_pct = 50.0

    # Use the absolute price values for range
    min_price = st.session_state.min_price_abs
    max_price = st.session_state.max_price_abs

    # Volatility range in decimal
    min_vol = st.session_state.min_vol_pct / 100
    max_vol = st.session_state.max_vol_pct / 100

    # Generate ranges
    prices = np.linspace(min_price, max_price, grid_size)
    volatilities = np.linspace(min_vol, max_vol, grid_size)

    # Calculate values for heatmap
    values = np.zeros((len(volatilities), len(prices)))

    for i, vol in enumerate(volatilities):
        for j, price in enumerate(prices):
            if metric == 'Price':
                values[i, j] = bs.option_price(price, K, T, r, vol, option_type.lower())
            elif metric == 'Delta':
                if option_type == 'Call':
                    values[i, j] = greeks.call_delta(price, K, T, r, vol)
                else:
                    values[i, j] = greeks.put_delta(price, K, T, r, vol)
            elif metric == 'Gamma':
                values[i, j] = greeks.gamma(price, K, T, r, vol)
            elif metric == 'Vega':
                values[i, j] = greeks.vega(price, K, T, r, vol)
            elif metric == 'Theta':
                if option_type == 'Call':
                    values[i, j] = greeks.call_theta(price, K, T, r, vol)
                else:
                    values[i, j] = greeks.put_theta(price, K, T, r, vol)
            elif metric == 'Rho':
                if option_type == 'Call':
                    values[i, j] = greeks.call_rho(price, K, T, r, vol)
                else:
                    values[i, j] = greeks.put_rho(price, K, T, r, vol)

    # Reverse for display (high volatility at top)
    values_display = values[::-1]
    vols_display = volatilities[::-1]

    # Create interactive heatmap
    fig = ui.create_interactive_heatmap(
        prices,
        vols_display,
        values_display,
        "Underlying Price ($)",
        "Volatility",
        f"{option_type} {metric} Heatmap"
    )

    st.plotly_chart(fig, width='stretch')

    # Heatmap Settings (below the chart)
    st.markdown("---")
    st.subheader("Heatmap Settings")

    # Initialize widget keys and tracking on first run
    if 'min_price_abs_input' not in st.session_state:
        st.session_state.min_price_abs_input = st.session_state.min_price_abs
    if 'min_price_pct_input' not in st.session_state:
        st.session_state.min_price_pct_input = st.session_state.min_price_pct
    if 'max_price_abs_input' not in st.session_state:
        st.session_state.max_price_abs_input = st.session_state.max_price_abs
    if 'max_price_pct_input' not in st.session_state:
        st.session_state.max_price_pct_input = st.session_state.max_price_pct

    # Track previous values to detect user changes
    if 'last_min_abs' not in st.session_state:
        st.session_state.last_min_abs = st.session_state.min_price_abs_input
    if 'last_min_pct' not in st.session_state:
        st.session_state.last_min_pct = st.session_state.min_price_pct_input
    if 'last_max_abs' not in st.session_state:
        st.session_state.last_max_abs = st.session_state.max_price_abs_input
    if 'last_max_pct' not in st.session_state:
        st.session_state.last_max_pct = st.session_state.max_price_pct_input

    # Sync inputs bidirectionally (use elif to prevent feedback loops)
    if abs(st.session_state.min_price_abs_input - st.session_state.last_min_abs) > 0.01:
        # Min absolute changed, update percentage
        st.session_state.min_price_pct_input = ((st.session_state.min_price_abs_input / S) - 1) * 100
        st.session_state.last_min_abs = st.session_state.min_price_abs_input
        st.session_state.last_min_pct = st.session_state.min_price_pct_input
    elif abs(st.session_state.min_price_pct_input - st.session_state.last_min_pct) > 0.01:
        # Min percentage changed, update absolute
        st.session_state.min_price_abs_input = S * (1 + st.session_state.min_price_pct_input / 100)
        st.session_state.last_min_pct = st.session_state.min_price_pct_input
        st.session_state.last_min_abs = st.session_state.min_price_abs_input

    if abs(st.session_state.max_price_abs_input - st.session_state.last_max_abs) > 0.01:
        # Max absolute changed, update percentage
        st.session_state.max_price_pct_input = ((st.session_state.max_price_abs_input / S) - 1) * 100
        st.session_state.last_max_abs = st.session_state.max_price_abs_input
        st.session_state.last_max_pct = st.session_state.max_price_pct_input
    elif abs(st.session_state.max_price_pct_input - st.session_state.last_max_pct) > 0.01:
        # Max percentage changed, update absolute
        st.session_state.max_price_abs_input = S * (1 + st.session_state.max_price_pct_input / 100)
        st.session_state.last_max_pct = st.session_state.max_price_pct_input
        st.session_state.last_max_abs = st.session_state.max_price_abs_input

    # Update main session state values for heatmap calculation
    st.session_state.min_price_abs = st.session_state.min_price_abs_input
    st.session_state.max_price_abs = st.session_state.max_price_abs_input

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Price Range**")

        col_a, col_b = st.columns(2)
        with col_a:
            st.number_input(
                "Min Price ($)",
                step=1.0,
                format="%.2f",
                key="min_price_abs_input"
            )

            st.number_input(
                "Min Change (%)",
                step=1.0,
                format="%.1f",
                key="min_price_pct_input"
            )

        with col_b:
            st.number_input(
                "Max Price ($)",
                step=1.0,
                format="%.2f",
                key="max_price_abs_input"
            )

            st.number_input(
                "Max Change (%)",
                step=1.0,
                format="%.1f",
                key="max_price_pct_input"
            )

    with col2:
        st.markdown("**Volatility Range**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.number_input(
                "Min Volatility (%)",
                value=st.session_state.min_vol_pct,
                min_value=0.1,
                step=1.0,
                format="%.1f",
                key="min_vol_pct_input"
            )
            st.session_state.min_vol_pct = st.session_state.min_vol_pct_input
        with col_b:
            st.number_input(
                "Max Volatility (%)",
                value=st.session_state.max_vol_pct,
                min_value=0.1,
                step=1.0,
                format="%.1f",
                key="max_vol_pct_input"
            )
            st.session_state.max_vol_pct = st.session_state.max_vol_pct_input

    # Download heatmap data
    st.markdown("---")
    heatmap_df = pd.DataFrame(
        values_display,
        index=[f"{v:.1%}" for v in vols_display],
        columns=[f"{p:.2f}" for p in prices]
    )
    ui.download_button_csv(heatmap_df, f"{option_type.lower()}_{metric.lower()}_heatmap.csv",
                           f"Download {metric} Heatmap Data")

# ============================================================================
# TAB 3: IMPLIED VOLATILITY
# ============================================================================
with tab3:
    st.header("Implied Volatility Calculator")

    st.markdown("""
    Enter a market option price to calculate the implied volatility.
    """)

    col1, col2 = st.columns(2)

    with col1:
        iv_option_type = st.radio("Option Type", ['Call', 'Put'], key='iv_type', horizontal=True)

    with col2:
        market_price = st.number_input(
            "Market Option Price ($)",
            min_value=0.01,
            value=10.0,
            step=0.5,
            help="Observed market price of the option"
        )

    if st.button("Calculate Implied Volatility", type="primary"):
        with st.spinner("Calculating..."):
            iv_result, success, message = iv_module.implied_volatility(
                market_price, S, K, T, r, iv_option_type.lower()
            )

        if success and iv_result is not None:
            st.success(f"Success: {message}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Implied Volatility", f"{iv_result*100:.2f}%")
            with col2:
                # Calculate theoretical price at this IV
                theo_price = bs.option_price(S, K, T, r, iv_result, iv_option_type.lower())
                st.metric("Theoretical Price", f"${theo_price:.4f}")
            with col3:
                diff = market_price - theo_price
                st.metric("Price Difference", f"${diff:.4f}")

            # Show comparison with current volatility assumption
            st.markdown("---")
            st.subheader("Volatility Comparison")
            comp_df = pd.DataFrame({
                'Metric': ['Volatility', 'Option Price'],
                'Market Implied': [f"{iv_result*100:.2f}%", f"${market_price:.4f}"],
                'Your Input': [f"{sigma*100:.2f}%", f"${bs.option_price(S, K, T, r, sigma, iv_option_type.lower()):.4f}"]
            })
            st.dataframe(comp_df, hide_index=True, width='stretch')

        else:
            st.error(f"Error: {message}")

# ============================================================================
# TAB 4: PAYOFF DIAGRAMS
# ============================================================================
with tab4:
    st.header("Payoff & Profit/Loss Diagrams")

    # Select option type and position
    col1, col2 = st.columns(2)
    with col1:
        payoff_option_type = st.selectbox("Option Type", ['Call', 'Put'])
    with col2:
        position = st.selectbox("Position", ['Long', 'Short'])

    # Calculate premium
    premium = bs.option_price(S, K, T, r, sigma, payoff_option_type.lower())

    st.info(f"Option Premium: ${premium:.4f}")

    # Generate price range
    S_T_range = payoffs.generate_price_range(S, K, num_points=200)

    # Calculate payoffs and profits
    if payoff_option_type == 'Call':
        if position == 'Long':
            profit = payoffs.long_call(S_T_range, K, premium)
            breakeven = [payoffs.calculate_breakeven_call(K, premium)]
        else:
            profit = payoffs.short_call(S_T_range, K, premium)
            breakeven = [payoffs.calculate_breakeven_call(K, premium)]
    else:  # Put
        if position == 'Long':
            profit = payoffs.long_put(S_T_range, K, premium)
            breakeven = [payoffs.calculate_breakeven_put(K, premium)]
        else:
            profit = payoffs.short_put(S_T_range, K, premium)
            breakeven = [payoffs.calculate_breakeven_put(K, premium)]

    # Create chart
    fig = ui.create_payoff_chart(
        S_T_range,
        profit,
        breakeven,
        title=f"{position} {payoff_option_type} Payoff Diagram (K=${K:.2f})"
    )

    st.plotly_chart(fig, width='stretch')

    # Display key metrics
    st.markdown("---")
    st.subheader("Position Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Premium", f"${premium:.4f}")

    with col2:
        if breakeven:
            st.metric("Break-Even", f"${breakeven[0]:.2f}")

    with col3:
        max_profit = np.max(profit)
        if max_profit > 1000:
            st.metric("Max Profit", "Unlimited" if position == 'Long' and payoff_option_type == 'Call' else f"${max_profit:.2f}")
        else:
            st.metric("Max Profit", f"${max_profit:.2f}")

    with col4:
        max_loss = np.min(profit)
        if abs(max_loss) > 1000:
            st.metric("Max Loss", "Unlimited" if position == 'Short' else f"${max_loss:.2f}")
        else:
            st.metric("Max Loss", f"${max_loss:.2f}")

# ============================================================================
# TAB 5: STRATEGY BUILDER
# ============================================================================
with tab5:
    st.header("Options Strategy Builder")

    strategy_type = st.selectbox(
        "Select Strategy",
        [
            "Long Straddle",
            "Short Straddle",
            "Long Strangle",
            "Short Strangle",
            "Bull Call Spread",
            "Bear Put Spread",
            "Iron Condor",
            "Custom"
        ]
    )

    if strategy_type != "Custom":
        # Predefined strategies
        if strategy_type == "Long Straddle":
            strategy = strategies.long_straddle(K, S, T, r, sigma)

        elif strategy_type == "Short Straddle":
            strategy = strategies.short_straddle(K, S, T, r, sigma)

        elif strategy_type == "Long Strangle":
            K_put = st.slider("Put Strike", float(S*0.7), float(S*0.95), float(S*0.9))
            K_call = st.slider("Call Strike", float(S*1.05), float(S*1.3), float(S*1.1))
            strategy = strategies.long_strangle(K_put, K_call, S, T, r, sigma)

        elif strategy_type == "Short Strangle":
            K_put = st.slider("Put Strike", float(S*0.7), float(S*0.95), float(S*0.9))
            K_call = st.slider("Call Strike", float(S*1.05), float(S*1.3), float(S*1.1))
            strategy = strategies.short_strangle(K_put, K_call, S, T, r, sigma)

        elif strategy_type == "Bull Call Spread":
            K_long = st.slider("Long Call Strike", float(S*0.9), float(S*1.1), float(S*0.95))
            K_short = st.slider("Short Call Strike", float(K_long), float(S*1.3), float(S*1.05))
            strategy = strategies.bull_call_spread(K_long, K_short, S, T, r, sigma)

        elif strategy_type == "Bear Put Spread":
            K_long = st.slider("Long Put Strike", float(S*0.9), float(S*1.1), float(S*1.05))
            K_short = st.slider("Short Put Strike", float(S*0.7), float(K_long), float(S*0.95))
            strategy = strategies.bear_put_spread(K_long, K_short, S, T, r, sigma)

        elif strategy_type == "Iron Condor":
            st.info("Iron Condor: Sell OTM put spread + Sell OTM call spread")
            K_put_long = st.slider("Long Put Strike (lowest)", float(S*0.7), float(S*0.9), float(S*0.85))
            K_put_short = st.slider("Short Put Strike", float(K_put_long), float(S*0.95), float(S*0.90))
            K_call_long = st.slider("Long Call Strike", float(S*1.05), float(S*1.3), float(S*1.15))
            K_call_short = st.slider("Short Call Strike (highest)", float(K_put_short), float(K_call_long), float(S*1.10))
            strategy = strategies.iron_condor(K_put_short, K_put_long, K_call_long, K_call_short, S, T, r, sigma)

        # Display strategy details
        st.subheader("Strategy Breakdown")

        legs_data = []
        for leg in strategy.legs:
            legs_data.append({
                'Type': leg.option_type.title(),
                'Position': leg.position.title(),
                'Strike': f"${leg.strike:.2f}",
                'Premium': f"${leg.premium:.4f}"
            })

        legs_df = pd.DataFrame(legs_data)
        st.dataframe(legs_df, hide_index=True, width='stretch')

        # Net premium
        net_premium = strategy.net_premium()
        if net_premium > 0:
            st.info(f"Net Premium Paid (Debit): ${net_premium:.4f}")
        else:
            st.success(f"Net Premium Received (Credit): ${-net_premium:.4f}")

        # Calculate strategy Greeks
        strategy_greeks = strategy.total_greeks(S, T, r, sigma)

        st.markdown("---")
        st.subheader("Strategy Greeks")

        cols = st.columns(5)
        greek_names = ['delta', 'gamma', 'vega', 'theta', 'rho']
        greek_labels = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']

        for col, name, label in zip(cols, greek_names, greek_labels):
            with col:
                st.metric(label, f"{strategy_greeks[name]:.4f}")

        # Payoff diagram
        st.markdown("---")
        st.subheader("Strategy Payoff Diagram")

        # Get all strikes for range calculation
        all_strikes = [leg.strike for leg in strategy.legs]
        S_T_range = payoffs.generate_price_range(S, all_strikes, num_points=300)

        # Calculate total payoff
        total_payoff = strategy.total_payoff(S_T_range)

        # Find breakeven points
        breakevens = payoffs.find_breakeven_points(S_T_range, total_payoff)

        # Create chart
        fig = ui.create_payoff_chart(
            S_T_range,
            total_payoff,
            breakevens,
            title=f"{strategy.name} Payoff Diagram"
        )

        st.plotly_chart(fig, width='stretch')

# ============================================================================
# TAB 6: MONTE CARLO SIMULATION
# ============================================================================
with tab6:
    st.header("Monte Carlo Simulation")

    st.markdown("""
    Price options using Monte Carlo simulation and compare with Black-Scholes.
    """)

    # Simulation parameters
    col1, col2 = st.columns(2)

    with col1:
        mc_option_type = st.radio("Option Type", ['Call', 'Put'], key='mc_type', horizontal=True)
        num_sims = st.select_slider(
            "Number of Simulations",
            options=[1000, 5000, 10000, 25000, 50000, 100000],
            value=10000
        )

    with col2:
        num_steps = st.slider("Time Steps", 50, 500, 252, step=50)
        random_seed = st.number_input("Random Seed (0 = random)", 0, 10000, 42, step=1)

    if st.button("Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            # Run simulation
            mc_stats = simulation.monte_carlo_statistics(
                S, K, T, r, sigma, mc_option_type.lower(),
                num_sims, num_steps, random_seed if random_seed > 0 else None
            )

            # Calculate Black-Scholes price for comparison
            bs_price = bs.option_price(S, K, T, r, sigma, mc_option_type.lower())

        # Display results
        st.success("Simulation Complete")

        st.markdown("---")
        st.subheader("Price Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Monte Carlo Price", f"${mc_stats['option_price']:.4f}")

        with col2:
            st.metric("Black-Scholes Price", f"${bs_price:.4f}")

        with col3:
            diff = mc_stats['option_price'] - bs_price
            diff_pct = (diff / bs_price) * 100
            st.metric("Difference", f"${diff:.4f}", f"{diff_pct:.2f}%")

        # Confidence interval
        ci_lower, ci_upper = mc_stats['confidence_interval_95']
        st.info(f"95% Confidence Interval: [${ci_lower:.4f}, ${ci_upper:.4f}]")

        # Terminal price distribution
        st.markdown("---")
        st.subheader("Terminal Price Distribution")

        fig = ui.create_distribution_chart(
            mc_stats['terminal_prices'],
            K,
            mc_option_type.lower(),
            f"Distribution of Stock Prices at Expiration ({num_sims:,} simulations)"
        )

        st.plotly_chart(fig, width='stretch')

        # Statistics
        st.markdown("---")
        st.subheader("Simulation Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("ITM Probability", f"{mc_stats['itm_probability']*100:.2f}%")

        with col2:
            st.metric("Mean Terminal Price", f"${mc_stats['terminal_price_mean']:.2f}")

        with col3:
            st.metric("Median Terminal Price", f"${mc_stats['terminal_price_median']:.2f}")

        with col4:
            st.metric("Std Dev", f"${mc_stats['terminal_price_std']:.2f}")

# ============================================================================
# TAB 7: SCENARIO ANALYSIS
# ============================================================================
with tab7:
    st.header("Scenario & Sensitivity Analysis")

    st.markdown("Analyze how option prices and Greeks change under different scenarios.")

    scenario_option_type = st.radio(
        "Option Type",
        ['Call', 'Put'],
        key='scenario_type',
        horizontal=True
    )

    st.markdown("---")
    st.subheader("Quick Scenarios")

    # Create scenario buttons
    col1, col2, col3, col4 = st.columns(4)

    scenarios = {}

    with col1:
        st.markdown("**Price Shocks**")
        if st.button("-10% Price"):
            scenarios['S'] = S * 0.9
        if st.button("-5% Price"):
            scenarios['S'] = S * 0.95
        if st.button("+5% Price"):
            scenarios['S'] = S * 1.05
        if st.button("+10% Price"):
            scenarios['S'] = S * 1.10

    with col2:
        st.markdown("**Volatility Shocks**")
        if st.button("÷2 Volatility"):
            scenarios['sigma'] = sigma * 0.5
        if st.button("-25% Vol"):
            scenarios['sigma'] = sigma * 0.75
        if st.button("+25% Vol"):
            scenarios['sigma'] = sigma * 1.25
        if st.button("×2 Volatility"):
            scenarios['sigma'] = sigma * 2.0

    with col3:
        st.markdown("**Time Decay**")
        if st.button("-7 Days"):
            scenarios['T'] = max(T - 7/365, 0.01)
        if st.button("-30 Days"):
            scenarios['T'] = max(T - 30/365, 0.01)
        if st.button("-90 Days"):
            scenarios['T'] = max(T - 90/365, 0.01)

    with col4:
        st.markdown("**Rate Changes**")
        if st.button("+1% Rate"):
            scenarios['r'] = r + 0.01
        if st.button("+2% Rate"):
            scenarios['r'] = r + 0.02

    # Apply scenario if any button was clicked
    if scenarios:
        S_scenario = scenarios.get('S', S)
        sigma_scenario = scenarios.get('sigma', sigma)
        T_scenario = scenarios.get('T', T)
        r_scenario = scenarios.get('r', r)

        # Calculate base case
        base_price = bs.option_price(S, K, T, r, sigma, scenario_option_type.lower())
        base_greeks = greeks.calculate_all_greeks(S, K, T, r, sigma, scenario_option_type.lower())

        # Calculate scenario case
        scenario_price = bs.option_price(S_scenario, K, T_scenario, r_scenario, sigma_scenario, scenario_option_type.lower())
        scenario_greeks = greeks.calculate_all_greeks(S_scenario, K, T_scenario, r_scenario, sigma_scenario, scenario_option_type.lower())

        st.markdown("---")
        st.subheader("Scenario Results")

        # Price comparison
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Base Price", f"${base_price:.4f}")

        with col2:
            st.metric("Scenario Price", f"${scenario_price:.4f}")

        with col3:
            price_change = scenario_price - base_price
            price_change_pct = (price_change / base_price) * 100
            st.metric("Change", f"${price_change:.4f}", f"{price_change_pct:.2f}%")

        # Greeks comparison
        st.markdown("---")
        st.subheader("Greeks Comparison")

        greeks_comparison = pd.DataFrame({
            'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
            'Base': [base_greeks[k] for k in ['delta', 'gamma', 'vega', 'theta', 'rho']],
            'Scenario': [scenario_greeks[k] for k in ['delta', 'gamma', 'vega', 'theta', 'rho']],
            'Change': [scenario_greeks[k] - base_greeks[k] for k in ['delta', 'gamma', 'vega', 'theta', 'rho']]
        })

        st.dataframe(greeks_comparison, hide_index=True, width='stretch')

    # Custom scenario inputs
    st.markdown("---")
    st.subheader("Custom Scenario")

    with st.expander("Define Custom Scenario"):
        col1, col2 = st.columns(2)

        with col1:
            custom_S = st.number_input("Stock Price", value=S, step=1.0, key='custom_S')
            custom_sigma = st.slider("Volatility", 0.01, 2.0, float(sigma), 0.01, key='custom_sigma')

        with col2:
            custom_T = st.slider("Time to Maturity", 0.01, 5.0, float(T), 0.01, key='custom_T')
            custom_r = st.slider("Risk-Free Rate", 0.0, 0.20, float(r), 0.01, key='custom_r', format="%.2f")

        if st.button("Apply Custom Scenario"):
            custom_price = bs.option_price(custom_S, K, custom_T, custom_r, custom_sigma, scenario_option_type.lower())
            st.success(f"Custom Scenario Price: ${custom_price:.4f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Black-Scholes Options Pricing Platform | Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)
