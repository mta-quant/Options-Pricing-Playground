"""
Option payoff and profit/loss calculations.

This module calculates payoffs at expiration and P&L including premiums paid.
"""

import numpy as np


def call_payoff(S_T, K):
    """
    Calculate call option payoff at expiration.

    Args:
        S_T: Stock price at expiration (can be array)
        K: Strike price

    Returns:
        Payoff (or array of payoffs)
    """
    return np.maximum(S_T - K, 0)


def put_payoff(S_T, K):
    """
    Calculate put option payoff at expiration.

    Args:
        S_T: Stock price at expiration (can be array)
        K: Strike price

    Returns:
        Payoff (or array of payoffs)
    """
    return np.maximum(K - S_T, 0)


def call_profit(S_T, K, premium):
    """
    Calculate call option profit/loss at expiration.

    Args:
        S_T: Stock price at expiration (can be array)
        K: Strike price
        premium: Premium paid for the option

    Returns:
        Profit/loss (or array)
    """
    return call_payoff(S_T, K) - premium


def put_profit(S_T, K, premium):
    """
    Calculate put option profit/loss at expiration.

    Args:
        S_T: Stock price at expiration (can be array)
        K: Strike price
        premium: Premium paid for the option

    Returns:
        Profit/loss (or array)
    """
    return put_payoff(S_T, K) - premium


def long_call(S_T, K, premium):
    """Long call position profit/loss."""
    return call_profit(S_T, K, premium)


def short_call(S_T, K, premium):
    """Short call position profit/loss."""
    return -call_profit(S_T, K, premium)


def long_put(S_T, K, premium):
    """Long put position profit/loss."""
    return put_profit(S_T, K, premium)


def short_put(S_T, K, premium):
    """Short put position profit/loss."""
    return -put_profit(S_T, K, premium)


def calculate_breakeven_call(K, premium):
    """
    Calculate break-even point for a long call.

    Args:
        K: Strike price
        premium: Premium paid

    Returns:
        float: Break-even stock price
    """
    return K + premium


def calculate_breakeven_put(K, premium):
    """
    Calculate break-even point for a long put.

    Args:
        K: Strike price
        premium: Premium paid

    Returns:
        float: Break-even stock price
    """
    return K - premium


def find_breakeven_points(S_T_range, profit_curve, tolerance=0.01):
    """
    Find break-even points in a profit curve.

    Args:
        S_T_range: Array of stock prices
        profit_curve: Array of profits at each price
        tolerance: How close to zero counts as break-even

    Returns:
        list: List of break-even prices
    """
    breakevens = []

    # Find zero crossings
    for i in range(len(profit_curve) - 1):
        if profit_curve[i] * profit_curve[i + 1] < 0:  # Sign change
            # Linear interpolation to find exact crossing
            x0, x1 = S_T_range[i], S_T_range[i + 1]
            y0, y1 = profit_curve[i], profit_curve[i + 1]
            breakeven = x0 - y0 * (x1 - x0) / (y1 - y0)
            breakevens.append(breakeven)

    return breakevens


def generate_price_range(S, K, num_points=200, range_multiplier=0.5):
    """
    Generate a reasonable price range for payoff charts.

    Args:
        S: Current stock price
        K: Strike price (or list of strikes)
        num_points: Number of points to generate
        range_multiplier: How far to extend range (as fraction of price)

    Returns:
        np.array: Array of stock prices
    """
    if isinstance(K, (list, np.ndarray)):
        min_K = min(K)
        max_K = max(K)
        center = (min_K + max_K) / 2
    else:
        center = K

    min_price = center * (1 - range_multiplier)
    max_price = center * (1 + range_multiplier)

    # Ensure current price is in range
    min_price = min(min_price, S * 0.5)
    max_price = max(max_price, S * 1.5)

    return np.linspace(min_price, max_price, num_points)
