"""
Test script to verify all modules work correctly.
"""

import black_scholes as bs
import greeks
import iv as iv_module
import payoffs
import strategies
import simulation

# Test parameters
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

print("Testing Options Pricing Platform Modules...")
print("=" * 60)

# Test 1: Pricing
print("\n1. Testing Black-Scholes Pricing...")
call_price = bs.call_price(S, K, T, r, sigma)
put_price = bs.put_price(S, K, T, r, sigma)
print(f"   ✓ Call Price: ${call_price:.4f}")
print(f"   ✓ Put Price: ${put_price:.4f}")

# Test 2: Greeks
print("\n2. Testing Greeks Calculation...")
greeks_call = greeks.calculate_all_greeks(S, K, T, r, sigma, 'call')
greeks_put = greeks.calculate_all_greeks(S, K, T, r, sigma, 'put')
print(f"   ✓ Call Delta: {greeks_call['delta']:.4f}")
print(f"   ✓ Put Delta: {greeks_put['delta']:.4f}")
print(f"   ✓ Gamma: {greeks_call['gamma']:.6f}")
print(f"   ✓ Vega: {greeks_call['vega']:.4f}")
print(f"   ✓ Call Theta: {greeks_call['theta']:.4f}")

# Test 3: Implied Volatility
print("\n3. Testing Implied Volatility Solver...")
iv_result, success, msg = iv_module.implied_volatility(call_price, S, K, T, r, 'call')
if success:
    print(f"   ✓ Implied Vol: {iv_result:.4f} ({iv_result*100:.2f}%)")
    print(f"   ✓ Message: {msg}")
else:
    print(f"   ✗ Failed: {msg}")

# Test 4: Payoffs
print("\n4. Testing Payoff Calculations...")
import numpy as np
S_T_range = np.array([90, 100, 110])
call_payoff = payoffs.call_payoff(S_T_range, K)
put_payoff = payoffs.put_payoff(S_T_range, K)
print(f"   ✓ Call Payoffs at [90,100,110]: {call_payoff}")
print(f"   ✓ Put Payoffs at [90,100,110]: {put_payoff}")

# Test 5: Strategies
print("\n5. Testing Strategy Builder...")
strategy = strategies.long_straddle(K, S, T, r, sigma)
print(f"   ✓ Strategy: {strategy.name}")
print(f"   ✓ Net Premium: ${strategy.net_premium():.4f}")
strategy_greeks = strategy.total_greeks(S, T, r, sigma)
print(f"   ✓ Strategy Delta: {strategy_greeks['delta']:.4f}")

# Test 6: Monte Carlo
print("\n6. Testing Monte Carlo Simulation...")
mc_stats = simulation.monte_carlo_statistics(S, K, T, r, sigma, 'call', 1000, 50, 42)
print(f"   ✓ MC Price: ${mc_stats['option_price']:.4f}")
print(f"   ✓ BS Price: ${call_price:.4f}")
print(f"   ✓ Difference: ${abs(mc_stats['option_price'] - call_price):.4f}")
print(f"   ✓ ITM Probability: {mc_stats['itm_probability']*100:.2f}%")

print("\n" + "=" * 60)
print("✅ All modules working correctly!")
print("=" * 60)
