"""
Implied Volatility calculator.

This module implements a robust Newton-Raphson solver with bisection fallback
for calculating implied volatility from market option prices.
"""

import numpy as np
from black_scholes import option_price
from greeks import vega


def implied_volatility(market_price, S, K, T, r, option_type='call',
                       initial_guess=0.3, tolerance=1e-6, max_iterations=100):
    """
    Calculate implied volatility using Newton-Raphson method with bisection fallback.

    Args:
        market_price: Observed market price of the option
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        option_type: 'call' or 'put'
        initial_guess: Starting volatility estimate
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations

    Returns:
        tuple: (implied_vol, success, message)
            - implied_vol: The calculated implied volatility (or None if failed)
            - success: Boolean indicating if calculation succeeded
            - message: Status message
    """
    # Validate inputs
    if market_price <= 0:
        return None, False, "Market price must be positive"

    if T <= 0:
        return None, False, "Time to maturity must be positive"

    # Check if option is within no-arbitrage bounds
    if option_type.lower() == 'call':
        intrinsic = max(S - K * np.exp(-r * T), 0)
        if market_price < intrinsic:
            return None, False, "Market price below intrinsic value (arbitrage)"
        if market_price > S:
            return None, False, "Call price cannot exceed stock price"
    else:  # put
        intrinsic = max(K * np.exp(-r * T) - S, 0)
        if market_price < intrinsic:
            return None, False, "Market price below intrinsic value (arbitrage)"
        if market_price > K * np.exp(-r * T):
            return None, False, "Put price cannot exceed discounted strike"

    # Try Newton-Raphson first
    sigma = initial_guess

    for i in range(max_iterations):
        # Calculate option price and vega at current sigma
        price = option_price(S, K, T, r, sigma, option_type)
        v = vega(S, K, T, r, sigma) * 100  # Convert back to decimal change

        # Check for convergence
        diff = price - market_price
        if abs(diff) < tolerance:
            return sigma, True, f"Converged in {i+1} iterations"

        # Check if vega is too small (Newton-Raphson will fail)
        if abs(v) < 1e-10:
            break

        # Newton-Raphson update
        sigma_new = sigma - diff / v

        # Ensure sigma stays positive
        if sigma_new <= 0:
            break

        # Check for oscillation
        if abs(sigma_new - sigma) < tolerance:
            return sigma_new, True, f"Converged in {i+1} iterations"

        sigma = sigma_new

    # If Newton-Raphson failed, try bisection method
    return _bisection_iv(market_price, S, K, T, r, option_type, tolerance, max_iterations)


def _bisection_iv(market_price, S, K, T, r, option_type, tolerance, max_iterations):
    """
    Bisection method for implied volatility (fallback method).

    Args:
        market_price: Observed market price
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        option_type: 'call' or 'put'
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations

    Returns:
        tuple: (implied_vol, success, message)
    """
    # Set bounds
    sigma_low = 0.001
    sigma_high = 5.0  # 500% volatility

    price_low = option_price(S, K, T, r, sigma_low, option_type)
    price_high = option_price(S, K, T, r, sigma_high, option_type)

    # Check if market price is within bounds
    if market_price < price_low:
        return None, False, "Market price too low for any reasonable volatility"
    if market_price > price_high:
        return None, False, "Market price too high for any reasonable volatility"

    # Bisection iterations
    for i in range(max_iterations):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = option_price(S, K, T, r, sigma_mid, option_type)

        diff = price_mid - market_price

        if abs(diff) < tolerance:
            return sigma_mid, True, f"Converged using bisection in {i+1} iterations"

        if diff > 0:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid

        # Check for convergence on sigma
        if abs(sigma_high - sigma_low) < tolerance:
            return sigma_mid, True, f"Converged using bisection in {i+1} iterations"

    # Didn't converge
    sigma_final = (sigma_low + sigma_high) / 2
    return sigma_final, False, f"Did not fully converge (best estimate: {sigma_final:.4f})"


def iv_surface(market_prices, S_values, K_values, T, r, option_type='call'):
    """
    Calculate implied volatility for a matrix of market prices.

    Args:
        market_prices: 2D array of market prices
        S_values: Array of underlying prices
        K_values: Array of strike prices
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        option_type: 'call' or 'put'

    Returns:
        tuple: (iv_matrix, success_matrix)
            - iv_matrix: 2D array of implied volatilities (NaN where failed)
            - success_matrix: 2D boolean array indicating success
    """
    n_strikes = len(K_values)
    n_spots = len(S_values)

    iv_matrix = np.full((n_strikes, n_spots), np.nan)
    success_matrix = np.zeros((n_strikes, n_spots), dtype=bool)

    for i, K in enumerate(K_values):
        for j, S in enumerate(S_values):
            market_price = market_prices[i, j]
            iv_result, success, _ = implied_volatility(
                market_price, S, K, T, r, option_type
            )
            if success and iv_result is not None:
                iv_matrix[i, j] = iv_result
                success_matrix[i, j] = True

    return iv_matrix, success_matrix
