"""
Option Greeks calculations.

This module implements all standard option Greeks:
Delta, Gamma, Vega, Theta, and Rho.
"""

import numpy as np
from scipy.stats import norm
from black_scholes import d1, d2


def call_delta(S, K, T, r, sigma):
    """
    Calculate delta for a call option.

    Delta measures the rate of change of option price with respect to
    the underlying asset price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: Call delta (0 to 1)
    """
    if T <= 0:
        return 1.0 if S > K else 0.0
    if sigma <= 0:
        return 1.0 if S > K else 0.0

    return norm.cdf(d1(S, K, T, r, sigma))


def put_delta(S, K, T, r, sigma):
    """
    Calculate delta for a put option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: Put delta (-1 to 0)
    """
    if T <= 0:
        return -1.0 if S < K else 0.0
    if sigma <= 0:
        return -1.0 if S < K else 0.0

    return call_delta(S, K, T, r, sigma) - 1


def gamma(S, K, T, r, sigma):
    """
    Calculate gamma (same for calls and puts).

    Gamma measures the rate of change of delta with respect to
    the underlying asset price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: Gamma
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d_1 = d1(S, K, T, r, sigma)
    return norm.pdf(d_1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    Calculate vega (same for calls and puts).

    Vega measures the sensitivity of option price to volatility changes.
    Note: Vega is returned per 1% change in volatility (divide by 100 for decimal).

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: Vega (per 1% volatility change)
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d_1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(d_1) * np.sqrt(T) / 100  # Divided by 100 for 1% change


def call_theta(S, K, T, r, sigma):
    """
    Calculate theta for a call option.

    Theta measures the rate of change of option price with respect to time.
    Returned as theta per day (divided by 365).

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: Call theta (per day)
    """
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return 0.0

    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    term1 = -(S * norm.pdf(d_1) * sigma) / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d_2)

    return (term1 + term2) / 365  # Per day


def put_theta(S, K, T, r, sigma):
    """
    Calculate theta for a put option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: Put theta (per day)
    """
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return 0.0

    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    term1 = -(S * norm.pdf(d_1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d_2)

    return (term1 + term2) / 365  # Per day


def call_rho(S, K, T, r, sigma):
    """
    Calculate rho for a call option.

    Rho measures the sensitivity of option price to interest rate changes.
    Returned per 1% change in interest rate.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: Call rho (per 1% rate change)
    """
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return 0.0

    d_2 = d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d_2) / 100  # Per 1% change


def put_rho(S, K, T, r, sigma):
    """
    Calculate rho for a put option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)

    Returns:
        float: Put rho (per 1% rate change)
    """
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return 0.0

    d_2 = d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d_2) / 100  # Per 1% change


def calculate_all_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate all Greeks for an option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'

    Returns:
        dict: Dictionary containing all Greeks
    """
    if option_type.lower() == 'call':
        return {
            'delta': call_delta(S, K, T, r, sigma),
            'gamma': gamma(S, K, T, r, sigma),
            'vega': vega(S, K, T, r, sigma),
            'theta': call_theta(S, K, T, r, sigma),
            'rho': call_rho(S, K, T, r, sigma)
        }
    elif option_type.lower() == 'put':
        return {
            'delta': put_delta(S, K, T, r, sigma),
            'gamma': gamma(S, K, T, r, sigma),
            'vega': vega(S, K, T, r, sigma),
            'theta': put_theta(S, K, T, r, sigma),
            'rho': put_rho(S, K, T, r, sigma)
        }
    else:
        raise ValueError("option_type must be 'call' or 'put'")
