# Black-Scholes Options Pricing & Analysis Platform

A comprehensive, production-quality options analysis tool built with Streamlit, featuring Black-Scholes pricing, Greeks calculation, implied volatility solving, strategy building, Monte Carlo simulation, and scenario analysis.

## Features

### ✅ Complete Feature Set

1. **Option Pricing & Greeks**
   - European call and put option pricing using Black-Scholes formula
   - Real-time calculation of all Greeks (Delta, Gamma, Vega, Theta, Rho)
   - Downloadable Greeks data (CSV)
   - Educational tooltips explaining each Greek

2. **Interactive Heatmaps**
   - Visualize option prices and Greeks across price/volatility ranges
   - User-selectable metrics (Price, Delta, Gamma, Vega, Theta, Rho)
   - Interactive Plotly charts with hover tooltips
   - Configurable grid resolution (10x10, 15x15, 20x20, 30x30)
   - Download heatmap data as CSV

3. **Implied Volatility Calculator**
   - Robust IV solver using Newton-Raphson with bisection fallback
   - Handles edge cases and validates no-arbitrage bounds
   - Compares market-implied vol with user input
   - Works for both calls and puts

4. **Payoff & P/L Diagrams**
   - Interactive payoff diagrams for any option position
   - Break-even point calculation and visualization
   - Max profit/loss metrics
   - Long and short positions for calls and puts

5. **Strategy Builder**
   - Pre-built strategies:
     - Long/Short Straddle
     - Long/Short Strangle
     - Bull Call Spread
     - Bear Put Spread
     - Iron Condor
   - Combined strategy Greeks
   - Net premium calculation (debit/credit)
   - Interactive payoff diagrams for multi-leg strategies
   - Custom strategy support

6. **Monte Carlo Simulation**
   - Price options using Geometric Brownian Motion (GBM)
   - Compare Monte Carlo price with Black-Scholes
   - Terminal price distribution visualization
   - ITM probability calculation
   - Confidence intervals
   - Configurable simulation parameters (paths, time steps, random seed)

7. **Scenario & Sensitivity Analysis**
   - Quick scenario buttons:
     - Price shocks (±5%, ±10%)
     - Volatility shocks (×0.5, ×2, ±25%)
     - Time decay (-7, -30, -90 days)
     - Rate changes (+1%, +2%)
   - Compare option prices and Greeks before/after scenarios
   - Custom scenario builder with full parameter control

## Project Structure

```
options-pricing-playground/
├── app.py                  # Main Streamlit application
├── black_scholes.py        # Black-Scholes pricing formulas
├── greeks.py               # Greeks calculations
├── iv.py                   # Implied volatility solver
├── payoffs.py              # Payoff and P&L functions
├── strategies.py           # Multi-leg strategy builder
├── simulation.py           # Monte Carlo simulation
├── ui_components.py        # Reusable UI components
├── requirements.txt        # Python dependencies
├── PYTHON_ENV_GUIDE.md     # Environment setup guide
└── README.md               # This file
```

## Installation & Setup

### Prerequisites

- Python 3.12+ (recommended)
- Miniforge/Mamba or conda

### Quick Start

1. **Activate your environment:**
   ```bash
   mamba activate opp
   ```

2. **Navigate to project directory:**
   ```bash
   cd ~/projects/options-pricing-playground
   ```

3. **Install dependencies** (if needed):
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to the URL displayed (typically `http://localhost:8501`)

## Usage Guide

### Basic Workflow

1. **Set Parameters** (Left Sidebar):
   - Underlying price (S)
   - Strike price (K) - use ATM/OTM/ITM presets
   - Time to expiration (T) - years or calendar date
   - Volatility (σ) - percentage
   - Risk-free rate (r) - percentage

2. **Explore Features** (Main Tabs):
   - **Pricing & Greeks**: View current prices and risk metrics
   - **Heatmaps**: Visualize sensitivities across ranges
   - **Implied Vol**: Reverse-engineer volatility from market prices
   - **Payoff Diagrams**: Understand position risk/reward
   - **Strategy Builder**: Create and analyze multi-leg strategies
   - **Monte Carlo**: Validate prices with simulation
   - **Scenario Analysis**: Test "what-if" scenarios

### Example Use Cases

#### Use Case 1: Analyze a Call Option
1. Set S=100, K=100, T=1 year, σ=20%, r=5%
2. View call price and Greeks in "Pricing & Greeks" tab
3. Check "Heatmaps" to see how price varies with spot/volatility
4. Use "Payoff Diagrams" to see break-even and max profit

#### Use Case 2: Build an Iron Condor
1. Go to "Strategy Builder" tab
2. Select "Iron Condor" from dropdown
3. Adjust strike prices with sliders
4. Review net credit received
5. Analyze combined Greeks
6. Study payoff diagram for profit zones

#### Use Case 3: Calculate Implied Volatility
1. Go to "Implied Volatility" tab
2. Enter market price of option (e.g., $10)
3. Select call or put
4. Click "Calculate Implied Volatility"
5. Compare with your volatility assumption

#### Use Case 4: Run Monte Carlo
1. Go to "Monte Carlo" tab
2. Select call or put
3. Choose number of simulations (10,000 recommended)
4. Click "Run Monte Carlo Simulation"
5. Compare with Black-Scholes price
6. View terminal price distribution

## Module Documentation

### black_scholes.py
Black-Scholes closed-form pricing formulas for European options.

**Key Functions:**
- `call_price(S, K, T, r, sigma)` - Calculate call option price
- `put_price(S, K, T, r, sigma)` - Calculate put option price
- `d1(S, K, T, r, sigma)` - Calculate d1 parameter
- `d2(S, K, T, r, sigma)` - Calculate d2 parameter

### greeks.py
Option Greeks (sensitivities) calculations.

**Key Functions:**
- `call_delta(...)` / `put_delta(...)` - Price sensitivity
- `gamma(...)` - Delta sensitivity (same for calls/puts)
- `vega(...)` - Volatility sensitivity (same for calls/puts)
- `call_theta(...)` / `put_theta(...)` - Time decay
- `call_rho(...)` / `put_rho(...)` - Rate sensitivity
- `calculate_all_greeks(...)` - Get all Greeks at once

### iv.py
Implied volatility solver with robust fallback methods.

**Key Functions:**
- `implied_volatility(market_price, S, K, T, r, option_type)` - Main IV solver
- Returns tuple: `(iv, success, message)`

### payoffs.py
Payoff and profit/loss calculations for options.

**Key Functions:**
- `call_payoff(S_T, K)` / `put_payoff(S_T, K)` - Payoff at expiration
- `long_call(...)` / `short_call(...)` / `long_put(...)` / `short_put(...)` - P&L functions
- `calculate_breakeven_call(...)` / `calculate_breakeven_put(...)` - Break-even points

### strategies.py
Multi-leg option strategies.

**Classes:**
- `OptionLeg` - Single option position
- `Strategy` - Multi-leg strategy container

**Strategy Builders:**
- `long_straddle(K, S, T, r, sigma)` - Buy call + put at same strike
- `long_strangle(K_put, K_call, ...)` - Buy OTM call + OTM put
- `bull_call_spread(K_long, K_short, ...)` - Bullish spread
- `bear_put_spread(K_long, K_short, ...)` - Bearish spread
- `iron_condor(...)` - Neutral strategy with defined risk

### simulation.py
Monte Carlo simulation for option pricing.

**Key Functions:**
- `simulate_gbm(S0, mu, sigma, T, num_simulations, num_steps)` - Generate price paths
- `monte_carlo_option_price(...)` - Price option via simulation
- `calculate_itm_probability(...)` - Probability of finishing ITM
- `monte_carlo_statistics(...)` - Comprehensive simulation stats

### ui_components.py
Reusable Streamlit UI components.

**Key Functions:**
- `create_parameter_inputs()` - Sidebar parameter controls
- `create_heatmap_controls()` - Heatmap configuration
- `display_option_prices(...)` - Price display widgets
- `display_greeks(...)` - Greeks table
- `create_interactive_heatmap(...)` - Plotly heatmap chart
- `create_payoff_chart(...)` - Plotly payoff diagram
- `download_button_csv(...)` - CSV download button

## Technical Details

### Black-Scholes Formula
The app implements the classic Black-Scholes-Merton formula for European options:

**Call Price:**
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
```

**Put Price:**
```
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
```

Where:
```
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Implied Volatility Algorithm
1. **Primary**: Newton-Raphson method using vega as derivative
2. **Fallback**: Bisection method if Newton-Raphson fails
3. **Validation**: Checks no-arbitrage bounds before solving

### Monte Carlo Method
Uses Geometric Brownian Motion (GBM) for stock price simulation:
```
S(t+Δt) = S(t) × exp[(r - σ²/2)Δt + σ√Δt × Z]
```
Where Z ~ N(0,1)

## Performance Considerations

- **Heatmaps**: Calculated on-demand; larger grids (30×30) may take 1-2 seconds
- **Monte Carlo**: 10,000 simulations typically complete in < 1 second
- **Implied Vol**: Converges in < 10 iterations for most cases
- **Strategy Builder**: Greeks calculated individually per leg, then summed

## Future Enhancements

Potential additions for future versions:

1. **American Options**: Binomial tree or finite difference methods
2. **Exotic Options**: Barriers, Asians, lookbacks
3. **Volatility Surface**: 3D visualization of IV across strikes/maturities
4. **Historical Data**: Integration with market data APIs
5. **Greeks Surface Plots**: 3D surfaces for each Greek
6. **Portfolio Mode**: Manage multiple positions
7. **PDF Reports**: Export analysis to PDF
8. **Real-time Streaming**: Live market data integration
9. **Black-Scholes Extensions**: Dividends, stock splits
10. **Machine Learning**: IV prediction, option pricing anomalies

## Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError"
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Heatmap takes too long
- **Solution**: Reduce grid resolution in sidebar

**Issue**: IV solver fails
- **Solution**: Check that market price is within no-arbitrage bounds

**Issue**: Monte Carlo gives different results each run
- **Solution**: Set a fixed random seed (non-zero value)

### Debug Mode
To enable Streamlit debug mode:
```bash
streamlit run app.py --logger.level=debug
```

## License

This project is open-source and available for educational and commercial use.

## Acknowledgments

Built using:
- **Streamlit** - Web framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **SciPy** - Statistical functions
- **Plotly** - Interactive visualizations

## Contact & Support

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Built with ❤️ using Streamlit and Python**
