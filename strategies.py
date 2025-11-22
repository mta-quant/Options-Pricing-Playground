"""
Option strategies builder and calculator.

This module defines common option strategies and calculates their
combined payoffs and Greeks.
"""

import numpy as np
from black_scholes import option_price
from greeks import calculate_all_greeks
from payoffs import long_call, short_call, long_put, short_put


class OptionLeg:
    """Represents a single option leg in a strategy."""

    def __init__(self, option_type, position, strike, premium=None):
        """
        Initialize an option leg.

        Args:
            option_type: 'call' or 'put'
            position: 'long' or 'short'
            strike: Strike price
            premium: Option premium (if None, will be calculated)
        """
        self.option_type = option_type.lower()
        self.position = position.lower()
        self.strike = strike
        self.premium = premium
        self.quantity = 1

    def calculate_premium(self, S, T, r, sigma):
        """Calculate premium using Black-Scholes if not provided."""
        if self.premium is None:
            self.premium = option_price(S, self.strike, T, r, sigma, self.option_type)
        return self.premium

    def payoff(self, S_T):
        """Calculate payoff at expiration."""
        if self.option_type == 'call':
            if self.position == 'long':
                return long_call(S_T, self.strike, self.premium) * self.quantity
            else:
                return short_call(S_T, self.strike, self.premium) * self.quantity
        else:  # put
            if self.position == 'long':
                return long_put(S_T, self.strike, self.premium) * self.quantity
            else:
                return short_put(S_T, self.strike, self.premium) * self.quantity

    def greeks(self, S, T, r, sigma):
        """Calculate Greeks for this leg."""
        greeks = calculate_all_greeks(S, self.strike, T, r, sigma, self.option_type)

        # Adjust sign for short positions
        multiplier = self.quantity if self.position == 'long' else -self.quantity

        return {k: v * multiplier for k, v in greeks.items()}


class Strategy:
    """Represents a multi-leg option strategy."""

    def __init__(self, name):
        """
        Initialize a strategy.

        Args:
            name: Strategy name (e.g., "Long Straddle")
        """
        self.name = name
        self.legs = []

    def add_leg(self, leg):
        """Add a leg to the strategy."""
        self.legs.append(leg)

    def calculate_all_premiums(self, S, T, r, sigma):
        """Calculate premiums for all legs."""
        for leg in self.legs:
            leg.calculate_premium(S, T, r, sigma)

    def net_premium(self):
        """
        Calculate net premium paid (positive) or received (negative).

        Returns:
            float: Net premium (positive = paid, negative = received)
        """
        net = 0
        for leg in self.legs:
            if leg.premium is not None:
                if leg.position == 'long':
                    net += leg.premium * leg.quantity
                else:
                    net -= leg.premium * leg.quantity
        return net

    def total_payoff(self, S_T):
        """
        Calculate total strategy payoff at expiration.

        Args:
            S_T: Stock price(s) at expiration

        Returns:
            Total payoff (array if S_T is array)
        """
        total = np.zeros_like(S_T, dtype=float)
        for leg in self.legs:
            total += leg.payoff(S_T)
        return total

    def total_greeks(self, S, T, r, sigma):
        """
        Calculate total Greeks for the strategy.

        Args:
            S: Current stock price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility

        Returns:
            dict: Combined Greeks
        """
        total_greeks = {
            'delta': 0,
            'gamma': 0,
            'vega': 0,
            'theta': 0,
            'rho': 0
        }

        for leg in self.legs:
            leg_greeks = leg.greeks(S, T, r, sigma)
            for greek in total_greeks:
                total_greeks[greek] += leg_greeks[greek]

        return total_greeks


# Strategy builders

def long_straddle(K, S, T, r, sigma):
    """
    Create a long straddle strategy.

    Buy a call and put at the same strike.

    Args:
        K: Strike price
        S: Current stock price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Strategy object
    """
    strategy = Strategy("Long Straddle")

    call_leg = OptionLeg('call', 'long', K)
    put_leg = OptionLeg('put', 'long', K)

    call_leg.calculate_premium(S, T, r, sigma)
    put_leg.calculate_premium(S, T, r, sigma)

    strategy.add_leg(call_leg)
    strategy.add_leg(put_leg)

    return strategy


def short_straddle(K, S, T, r, sigma):
    """Create a short straddle strategy."""
    strategy = Strategy("Short Straddle")

    call_leg = OptionLeg('call', 'short', K)
    put_leg = OptionLeg('put', 'short', K)

    call_leg.calculate_premium(S, T, r, sigma)
    put_leg.calculate_premium(S, T, r, sigma)

    strategy.add_leg(call_leg)
    strategy.add_leg(put_leg)

    return strategy


def long_strangle(K_put, K_call, S, T, r, sigma):
    """
    Create a long strangle strategy.

    Buy a put at lower strike and call at higher strike.

    Args:
        K_put: Put strike price (lower)
        K_call: Call strike price (higher)
        S: Current stock price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Strategy object
    """
    strategy = Strategy("Long Strangle")

    put_leg = OptionLeg('put', 'long', K_put)
    call_leg = OptionLeg('call', 'long', K_call)

    put_leg.calculate_premium(S, T, r, sigma)
    call_leg.calculate_premium(S, T, r, sigma)

    strategy.add_leg(put_leg)
    strategy.add_leg(call_leg)

    return strategy


def short_strangle(K_put, K_call, S, T, r, sigma):
    """Create a short strangle strategy."""
    strategy = Strategy("Short Strangle")

    put_leg = OptionLeg('put', 'short', K_put)
    call_leg = OptionLeg('call', 'short', K_call)

    put_leg.calculate_premium(S, T, r, sigma)
    call_leg.calculate_premium(S, T, r, sigma)

    strategy.add_leg(put_leg)
    strategy.add_leg(call_leg)

    return strategy


def bull_call_spread(K_long, K_short, S, T, r, sigma):
    """
    Create a bull call spread.

    Buy call at lower strike, sell call at higher strike.

    Args:
        K_long: Long call strike (lower)
        K_short: Short call strike (higher)
        S: Current stock price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Strategy object
    """
    strategy = Strategy("Bull Call Spread")

    long_leg = OptionLeg('call', 'long', K_long)
    short_leg = OptionLeg('call', 'short', K_short)

    long_leg.calculate_premium(S, T, r, sigma)
    short_leg.calculate_premium(S, T, r, sigma)

    strategy.add_leg(long_leg)
    strategy.add_leg(short_leg)

    return strategy


def bear_put_spread(K_long, K_short, S, T, r, sigma):
    """
    Create a bear put spread.

    Buy put at higher strike, sell put at lower strike.

    Args:
        K_long: Long put strike (higher)
        K_short: Short put strike (lower)
        S: Current stock price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Strategy object
    """
    strategy = Strategy("Bear Put Spread")

    long_leg = OptionLeg('put', 'long', K_long)
    short_leg = OptionLeg('put', 'short', K_short)

    long_leg.calculate_premium(S, T, r, sigma)
    short_leg.calculate_premium(S, T, r, sigma)

    strategy.add_leg(long_leg)
    strategy.add_leg(short_leg)

    return strategy


def iron_condor(K_put_short, K_put_long, K_call_long, K_call_short, S, T, r, sigma):
    """
    Create an iron condor.

    Sell put spread and call spread.

    Args:
        K_put_short: Short put strike (higher put strike)
        K_put_long: Long put strike (lower put strike)
        K_call_long: Long call strike (lower call strike)
        K_call_short: Short call strike (higher call strike)
        S: Current stock price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Strategy object
    """
    strategy = Strategy("Iron Condor")

    # Put spread (bear put spread - sold)
    put_long = OptionLeg('put', 'long', K_put_long)
    put_short = OptionLeg('put', 'short', K_put_short)

    # Call spread (bull call spread - sold)
    call_long = OptionLeg('call', 'long', K_call_long)
    call_short = OptionLeg('call', 'short', K_call_short)

    put_long.calculate_premium(S, T, r, sigma)
    put_short.calculate_premium(S, T, r, sigma)
    call_long.calculate_premium(S, T, r, sigma)
    call_short.calculate_premium(S, T, r, sigma)

    strategy.add_leg(put_long)
    strategy.add_leg(put_short)
    strategy.add_leg(call_long)
    strategy.add_leg(call_short)

    return strategy
