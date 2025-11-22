"""
Black-Scholes option pricing formulas.

This module implements the closed-form Black-Scholes-Merton formulas
for European call and put options.
"""

import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    """
    Calculate d1 parameter in Black-Scholes formula.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: d1 value
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    """
    Calculate d2 parameter in Black-Scholes formula.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: d2 value
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    """
    Calculate European call option price using Black-Scholes formula.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: Call option price
    """
    if T <= 0:
        return max(S - K, 0)
    if sigma <= 0:
        return max(S - K * np.exp(-r * T), 0)

    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    price = S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    return price


def put_price(S, K, T, r, sigma):
    """
    Calculate European put option price using Black-Scholes formula.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: Put option price
    """
    if T <= 0:
        return max(K - S, 0)
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S, 0)

    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    price = K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)
    return price


def option_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option price for either call or put.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'

    Returns:
        float: Option price
    """
    if option_type.lower() == 'call':
        return call_price(S, K, T, r, sigma)
    elif option_type.lower() == 'put':
        return put_price(S, K, T, r, sigma)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
