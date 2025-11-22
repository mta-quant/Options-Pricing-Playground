"""
Monte Carlo simulation for option pricing.

This module implements Geometric Brownian Motion simulation
for option pricing and analysis.
"""

import numpy as np
from payoffs import call_payoff, put_payoff


def simulate_gbm(S0, mu, sigma, T, num_simulations=10000, num_steps=252, random_seed=None):
    """
    Simulate stock price paths using Geometric Brownian Motion.

    Args:
        S0: Initial stock price
        mu: Drift (expected return, typically r for risk-neutral)
        sigma: Volatility (annualized)
        T: Time horizon (years)
        num_simulations: Number of paths to simulate
        num_steps: Number of time steps
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (paths, time_grid)
            - paths: Array of shape (num_simulations, num_steps+1) with price paths
            - time_grid: Array of time points
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / num_steps
    time_grid = np.linspace(0, T, num_steps + 1)

    # Generate random normal increments
    Z = np.random.standard_normal((num_simulations, num_steps))

    # Initialize paths array
    paths = np.zeros((num_simulations, num_steps + 1))
    paths[:, 0] = S0

    # Simulate paths using exact solution of GBM
    for t in range(1, num_steps + 1):
        paths[:, t] = paths[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
        )

    return paths, time_grid


def monte_carlo_option_price(S, K, T, r, sigma, option_type='call',
                              num_simulations=10000, num_steps=252, random_seed=None):
    """
    Price an option using Monte Carlo simulation.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'
        num_simulations: Number of simulations
        num_steps: Number of time steps
        random_seed: Random seed for reproducibility

    Returns:
        dict: Dictionary with price, std_error, and terminal prices
    """
    # Simulate stock price paths (risk-neutral: mu = r)
    paths, _ = simulate_gbm(S, r, sigma, T, num_simulations, num_steps, random_seed)

    # Terminal stock prices
    S_T = paths[:, -1]

    # Calculate payoffs at expiration
    if option_type.lower() == 'call':
        payoffs = call_payoff(S_T, K)
    elif option_type.lower() == 'put':
        payoffs = put_payoff(S_T, K)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discount back to present value
    discounted_payoffs = payoffs * np.exp(-r * T)

    # Calculate option price and standard error
    option_price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)

    return {
        'price': option_price,
        'std_error': std_error,
        'terminal_prices': S_T,
        'payoffs': payoffs,
        'confidence_interval_95': (
            option_price - 1.96 * std_error,
            option_price + 1.96 * std_error
        )
    }


def calculate_itm_probability(S, K, T, r, sigma, option_type='call',
                               num_simulations=10000, num_steps=252, random_seed=None):
    """
    Calculate probability of finishing in-the-money using Monte Carlo.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'
        num_simulations: Number of simulations
        num_steps: Number of time steps
        random_seed: Random seed

    Returns:
        float: Probability of finishing ITM (0 to 1)
    """
    # Simulate stock price paths
    paths, _ = simulate_gbm(S, r, sigma, T, num_simulations, num_steps, random_seed)

    # Terminal stock prices
    S_T = paths[:, -1]

    # Calculate ITM probability
    if option_type.lower() == 'call':
        itm = S_T > K
    elif option_type.lower() == 'put':
        itm = S_T < K
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return np.mean(itm)


def monte_carlo_statistics(S, K, T, r, sigma, option_type='call',
                            num_simulations=10000, num_steps=252, random_seed=None):
    """
    Calculate comprehensive Monte Carlo statistics for an option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'
        num_simulations: Number of simulations
        num_steps: Number of time steps
        random_seed: Random seed

    Returns:
        dict: Comprehensive statistics
    """
    # Run Monte Carlo pricing
    mc_result = monte_carlo_option_price(
        S, K, T, r, sigma, option_type, num_simulations, num_steps, random_seed
    )

    # Calculate ITM probability
    itm_prob = calculate_itm_probability(
        S, K, T, r, sigma, option_type, num_simulations, num_steps, random_seed
    )

    # Terminal price statistics
    S_T = mc_result['terminal_prices']

    stats = {
        'option_price': mc_result['price'],
        'std_error': mc_result['std_error'],
        'confidence_interval_95': mc_result['confidence_interval_95'],
        'itm_probability': itm_prob,
        'terminal_price_mean': np.mean(S_T),
        'terminal_price_median': np.median(S_T),
        'terminal_price_std': np.std(S_T),
        'terminal_price_min': np.min(S_T),
        'terminal_price_max': np.max(S_T),
        'terminal_prices': S_T,
        'payoffs': mc_result['payoffs']
    }

    return stats


def convergence_analysis(S, K, T, r, sigma, option_type='call',
                         max_simulations=50000, step=1000, random_seed=None):
    """
    Analyze convergence of Monte Carlo price estimate.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)
        option_type: 'call' or 'put'
        max_simulations: Maximum number of simulations
        step: Step size for convergence analysis
        random_seed: Random seed

    Returns:
        tuple: (n_sims_array, prices_array, errors_array)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Simulate all paths at once
    paths, _ = simulate_gbm(S, r, sigma, T, max_simulations, 252, random_seed)
    S_T = paths[:, -1]

    # Calculate all payoffs
    if option_type.lower() == 'call':
        payoffs = call_payoff(S_T, K)
    else:
        payoffs = put_payoff(S_T, K)

    discounted_payoffs = payoffs * np.exp(-r * T)

    # Calculate running averages
    n_sims_array = range(step, max_simulations + 1, step)
    prices_array = []
    errors_array = []

    for n in n_sims_array:
        price = np.mean(discounted_payoffs[:n])
        error = np.std(discounted_payoffs[:n]) / np.sqrt(n)
        prices_array.append(price)
        errors_array.append(error)

    return np.array(n_sims_array), np.array(prices_array), np.array(errors_array)
