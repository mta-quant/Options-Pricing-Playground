# Quick Start Guide

## Running the App

```bash
# 1. Activate environment
mamba activate opp

# 2. Navigate to project
cd ~/projects/options-pricing-playground

# 3. Run the app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Project Files

```
options-pricing-playground/
├── app.py                  ← Main Streamlit app (START HERE)
├── black_scholes.py        ← Pricing formulas
├── greeks.py               ← Greeks calculations
├── iv.py                   ← Implied volatility solver
├── payoffs.py              ← Payoff functions
├── strategies.py           ← Strategy builder
├── simulation.py           ← Monte Carlo
├── ui_components.py        ← UI helpers
├── test_modules.py         ← Run tests
├── requirements.txt        ← Dependencies
├── README.md               ← Full documentation
├── PYTHON_ENV_GUIDE.md     ← Environment setup
└── QUICK_START.md          ← This file
```

## Quick Test

Verify everything works:
```bash
python test_modules.py
```

You should see:
```
✅ All modules working correctly!
```

## Tab Overview

| Tab | Purpose |
|-----|---------|
| 💰 **Pricing & Greeks** | View option prices and risk metrics |
| 📊 **Heatmaps** | Visualize sensitivities across price/vol ranges |
| 🔍 **Implied Volatility** | Calculate IV from market prices |
| 📉 **Payoff Diagrams** | See profit/loss at expiration |
| 🎯 **Strategy Builder** | Create multi-leg strategies |
| 🎲 **Monte Carlo** | Simulate option pricing |
| ⚡ **Scenario Analysis** | Test what-if scenarios |

## Common Tasks

### Price an Option
1. Set parameters in sidebar (S, K, T, r, σ)
2. View results in "Pricing & Greeks" tab

### Build a Straddle
1. Go to "Strategy Builder" tab
2. Select "Long Straddle"
3. View combined payoff and Greeks

### Calculate Implied Vol
1. Go to "Implied Volatility" tab
2. Enter market option price
3. Click "Calculate"

### Run Monte Carlo
1. Go to "Monte Carlo" tab
2. Set simulation parameters
3. Click "Run Monte Carlo Simulation"

## Keyboard Shortcuts

While app is focused:
- `R` - Rerun the app
- `C` - Clear cache
- `?` - Show keyboard shortcuts

## Troubleshooting

**App won't start?**
```bash
pip install -r requirements.txt
```

**Module errors?**
```bash
python test_modules.py
```

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

## Need Help?

- Full docs: See `README.md`
- Environment setup: See `PYTHON_ENV_GUIDE.md`
- Test modules: Run `python test_modules.py`

---

**Ready to go! Run `streamlit run app.py` to start.**
