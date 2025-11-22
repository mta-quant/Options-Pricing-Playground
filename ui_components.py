"""
Reusable UI components for Streamlit app.

This module contains helper functions for creating consistent UI elements.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io


def create_parameter_inputs(defaults=None):
    """
    Create sidebar inputs for option parameters.

    Args:
        defaults: Dictionary with default values

    Returns:
        dict: Dictionary with all parameter values
    """
    if defaults is None:
        defaults = {
            'S': 100.0,
            'K': 100.0,
            'r': 5.0,
            'T': 1.0,
            'sigma': 0.2
        }

    st.sidebar.header("📊 Option Parameters")

    # Underlying price
    with st.sidebar.expander("🎯 Underlying Price", expanded=True):
        S = st.number_input(
            "Current Stock Price ($)",
            min_value=0.01,
            value=defaults['S'],
            step=1.0,
            format="%.2f",
            help="Current market price of the underlying asset"
        )

    # Strike and moneyness presets
    with st.sidebar.expander("🎲 Strike Price", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("ATM", help="At-the-money"):
                defaults['K'] = S
        with col2:
            if st.button("10% OTM", help="10% out-of-the-money"):
                defaults['K'] = S * 1.10
        with col3:
            if st.button("10% ITM", help="10% in-the-money"):
                defaults['K'] = S * 0.90

        K = st.number_input(
            "Strike Price ($)",
            min_value=0.01,
            value=defaults['K'],
            step=1.0,
            format="%.2f",
            help="Option strike price"
        )

    # Time to maturity
    with st.sidebar.expander("📅 Time to Expiration", expanded=True):
        time_input_method = st.radio(
            "Input Method",
            ["Years", "Calendar Date"],
            help="Choose how to specify expiration"
        )

        if time_input_method == "Years":
            T = st.slider(
                "Time to Maturity (Years)",
                min_value=0.01,
                max_value=5.0,
                value=defaults['T'],
                step=0.01,
                help="Time until option expiration in years"
            )
        else:
            expiry_date = st.date_input(
                "Expiration Date",
                value=datetime.now() + timedelta(days=365),
                min_value=datetime.now(),
                help="Calendar date of option expiration"
            )
            days_to_expiry = (expiry_date - datetime.now().date()).days
            T = max(days_to_expiry / 365.0, 0.01)
            st.info(f"⏳ Time to expiration: {T:.4f} years ({days_to_expiry} days)")

    # Volatility
    with st.sidebar.expander("📈 Volatility", expanded=True):
        sigma_pct = st.slider(
            "Volatility (%)",
            min_value=1.0,
            max_value=200.0,
            value=defaults['sigma'] * 100,
            step=1.0,
            help="Annualized volatility (standard deviation)"
        )
        sigma = sigma_pct / 100.0

    # Risk-free rate
    with st.sidebar.expander("💰 Risk-Free Rate", expanded=True):
        r_pct = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=defaults['r'],
            step=0.1,
            help="Annualized risk-free interest rate"
        )
        r = r_pct / 100.0

    return {
        'S': S,
        'K': K,
        'T': T,
        'r': r,
        'sigma': sigma,
        'r_pct': r_pct,
        'sigma_pct': sigma_pct
    }


def create_heatmap_controls():
    """
    Create controls for heatmap settings.

    Returns:
        dict: Heatmap configuration
    """
    st.sidebar.markdown("---")
    st.sidebar.header("🗺️ Heatmap Settings")

    with st.sidebar.expander("Price Range", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min Price", value=80.0, step=5.0)
        with col2:
            max_price = st.number_input("Max Price", value=120.0, step=5.0)

    with st.sidebar.expander("Volatility Range", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            min_vol = st.number_input("Min Vol", value=0.1, step=0.05, format="%.2f")
        with col2:
            max_vol = st.number_input("Max Vol", value=0.5, step=0.05, format="%.2f")

    with st.sidebar.expander("Grid Resolution", expanded=True):
        grid_size = st.selectbox(
            "Grid Size",
            options=[10, 15, 20, 30],
            index=1,
            help="Number of grid points in each dimension"
        )

    return {
        'min_price': min_price,
        'max_price': max_price,
        'min_vol': min_vol,
        'max_vol': max_vol,
        'grid_size': grid_size
    }


def display_option_prices(call_price, put_price):
    """Display current option prices in a nice format."""
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="📞 Call Option Price",
            value=f"${call_price:.4f}",
            help="European call option price"
        )

    with col2:
        st.metric(
            label="📉 Put Option Price",
            value=f"${put_price:.4f}",
            help="European put option price"
        )


def display_greeks(greeks_call, greeks_put):
    """
    Display Greeks in a formatted table.

    Args:
        greeks_call: Dictionary of call Greeks
        greeks_put: Dictionary of put Greeks
    """
    st.subheader("📊 Option Greeks")

    greeks_df = pd.DataFrame({
        'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
        'Call': [
            f"{greeks_call['delta']:.4f}",
            f"{greeks_call['gamma']:.6f}",
            f"{greeks_call['vega']:.4f}",
            f"{greeks_call['theta']:.4f}",
            f"{greeks_call['rho']:.4f}"
        ],
        'Put': [
            f"{greeks_put['delta']:.4f}",
            f"{greeks_put['gamma']:.6f}",
            f"{greeks_put['vega']:.4f}",
            f"{greeks_put['theta']:.4f}",
            f"{greeks_put['rho']:.4f}"
        ],
        'Description': [
            'Sensitivity to price ($)',
            'Rate of change of delta',
            'Sensitivity to volatility (per 1%)',
            'Time decay (per day)',
            'Sensitivity to interest rate (per 1%)'
        ]
    })

    st.dataframe(greeks_df, use_container_width=True, hide_index=True)


def create_interactive_heatmap(x_values, y_values, z_values, x_label, y_label, title, colorscale='RdYlGn'):
    """
    Create an interactive heatmap using Plotly.

    Args:
        x_values: X-axis values
        y_values: Y-axis values
        z_values: Z-values (2D array)
        x_label: X-axis label
        y_label: Y-axis label
        title: Chart title
        colorscale: Plotly colorscale

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        x=x_values,
        y=y_values,
        z=z_values,
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate=f'{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<br>Value: %{{z:.4f}}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        hovermode='closest'
    )

    return fig


def create_payoff_chart(S_T_range, payoff, breakeven_points=None, title="Payoff Diagram"):
    """
    Create a payoff/P&L chart using Plotly.

    Args:
        S_T_range: Array of stock prices
        payoff: Array of payoffs/profits
        breakeven_points: List of breakeven prices (optional)
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Add payoff curve
    fig.add_trace(go.Scatter(
        x=S_T_range,
        y=payoff,
        mode='lines',
        name='P&L',
        line=dict(color='blue', width=2),
        hovertemplate='Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Add breakeven points
    if breakeven_points:
        for be in breakeven_points:
            fig.add_vline(
                x=be,
                line_dash="dot",
                line_color="red",
                opacity=0.7,
                annotation_text=f"BE: ${be:.2f}",
                annotation_position="top"
            )

    # Color areas above/below zero
    fig.add_trace(go.Scatter(
        x=S_T_range,
        y=payoff,
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(0, 255, 0, 0.2)',
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="Profit/Loss ($)",
        height=500,
        hovermode='x unified',
        showlegend=True
    )

    return fig


def create_distribution_chart(terminal_prices, K, option_type, title="Terminal Price Distribution"):
    """
    Create a histogram of terminal prices from Monte Carlo.

    Args:
        terminal_prices: Array of simulated terminal prices
        K: Strike price
        option_type: 'call' or 'put'
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Create histogram
    fig.add_trace(go.Histogram(
        x=terminal_prices,
        nbinsx=50,
        name='Terminal Prices',
        marker_color='lightblue',
        opacity=0.7
    ))

    # Add strike line
    fig.add_vline(
        x=K,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Strike: ${K:.2f}",
        annotation_position="top right"
    )

    # Shade ITM region
    if option_type.lower() == 'call':
        itm_region = terminal_prices[terminal_prices > K]
        label = "ITM (> Strike)"
    else:
        itm_region = terminal_prices[terminal_prices < K]
        label = "ITM (< Strike)"

    fig.update_layout(
        title=title,
        xaxis_title="Stock Price ($)",
        yaxis_title="Frequency",
        height=400,
        showlegend=True
    )

    return fig


def download_button_csv(df, filename, label="Download CSV"):
    """
    Create a download button for CSV data.

    Args:
        df: Pandas DataFrame
        filename: Download filename
        label: Button label
    """
    csv = df.to_csv(index=True)
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime='text/csv'
    )


def create_greeks_surface_data(S_range, sigma_range, K, T, r, greek_func, option_type):
    """
    Generate data for Greek surface plots.

    Args:
        S_range: Array of stock prices
        sigma_range: Array of volatilities
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        greek_func: Function to calculate Greek
        option_type: 'call' or 'put'

    Returns:
        2D array of Greek values
    """
    n_sigma = len(sigma_range)
    n_S = len(S_range)

    greek_values = np.zeros((n_sigma, n_S))

    for i, sigma in enumerate(sigma_range):
        for j, S in enumerate(S_range):
            greek_values[i, j] = greek_func(S, K, T, r, sigma)

    return greek_values
