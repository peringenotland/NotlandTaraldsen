# ------------------------------------------------------------
# SM_2000.py
# ------------------------------------------------------------
# This script is made to reproduce the results from the paper:
# Schwartz, Moon (2000), Rational pricing of internet companies.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
gvkey = 225094 # Vestas

R_0 = 356  # Initial revenue in millions per quarter
# R_0 = parameters.get_initial_revenue(gvkey)

L_0 = 559 # Loss-carryforward in millions
X_0 = 906 # Initial cash balance in millions
mu_0 = 0.11  # Initial growth rate per quarter
sigma_0 = 0.10  # Initial revenue volatility per quarter
eta_0 = 0.03 # Initial volatility of expected growth rate
rho = 0.0 # Correlation between revenue and growth rate
mu_mean = 0.015  # Mean-reversion level for growth rate
sigma_mean = 0.05  # Mean-reversion level for volatility
taxrate = 0.35  # Corporate tax rate
r_f = 0.05  # Risk-free rate
kappa_mu = 0.07  # Mean-reversion speed for expected growth rate
kappa_sigma = 0.07  # Mean-reversion speed for volatility
kappa_eta = 0.07  # Mean-reversion speed for expected growth rate volatility
alpha = 0.75 # COGS as a part of revenues
F = 75 # Fixed costs in millions per quarter
beta = 0.19 # Variable component of other expenses
lambda_1 = 0.01 # Market price of risk for the revenue factor
lambda_2 = 0.0 # Market price of risk for the expected rate of growth in revenues factor
T = 25  # Time horizon in years
dt = 1 # Time step
M = 10 # Exit multiple
simulations = 100000  # Number of Monte Carlo runs


# Simulation setup
num_steps = T * 4 + 1 # Quarters in 25 years + initial step
np.random.seed(42) # Seed random number generator

R = np.zeros((simulations, num_steps)) # Revenue trajectories
R_real = np.zeros((simulations, num_steps)) # Revenue trajectories
X = np.zeros((simulations, num_steps)) # Cash balance trajectories
mu = np.zeros((simulations, num_steps)) # Growth rate trajectories
mu_real = np.zeros((simulations, num_steps)) # Growth rate trajectories
sigma = np.zeros(num_steps)
eta = np.zeros(num_steps)
# L = np.zeros((simulations, num_steps)) # Loss carry-forward trajectories
L = np.zeros((simulations, T+1), dtype=np.float32)  # Loss carryforward tracked yearly
bankruptcy = np.zeros((simulations, num_steps), dtype=bool) # Bankruptcy indicator
EBIT = np.zeros((simulations, num_steps))  # Earnings before interest and taxes


# Initial values (t = 0) not simulated
R[:, 0] = R_0 # Initial revenue
R_real[:, 0] = R_0 # Initial revenue physical
X[:, 0] = X_0 # Initial cash balance
mu[:, 0] = mu_0 # Initial growth rate
mu_real[:, 0] = mu_0 # Initial growth rate physical
sigma[0] = sigma_0
eta[0] = eta_0
L[:, 0] = L_0 # Initial loss carry-forward
bankruptcy[:, 0] = False # Initial bankruptcy indicator

# Generate correlated random shocks
Z1 = np.random.randn(simulations, num_steps)  # Standard normal noise for revenue
Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(simulations, num_steps)  # Correlated noise for growth


# Monte Carlo simulation
for t in range(1, num_steps):

    # Only update non-bankrupt firms
    active_firms = ~bankruptcy[:, t-1]  # Firms that haven't gone bankrupt
    
    # Update growth rate with mean-reversion
    mu[:, t] = np.exp(-kappa_mu * dt) * mu[:, t-1] + (1 - np.exp(-kappa_mu * dt)) * (mu_mean - (lambda_2*eta[t-1])/(kappa_mu)) + np.sqrt((1 - np.exp(-2*kappa_mu*dt))/(2*kappa_mu)) * eta[t-1] * np.sqrt(dt) * Z2[:, t] # Good med eq18, SM2000
    
    # mu_real[:, t] = np.exp(-kappa_mu * dt) * mu[:, t-1] + (1 - np.exp(-kappa_mu * dt)) * (mu_mean - (lambda_2*eta[t-1])/(kappa_mu)) + np.sqrt((1 - np.exp(-2*kappa_mu*dt))/(2*kappa_mu)) * eta[t-1] * np.sqrt(dt) * Z2[:, t] # Good med eq18, SM2000


    sigma[t] = sigma[0] * np.exp(-kappa_sigma * t) + sigma_mean * (1 - np.exp(-kappa_sigma * t)) # Good med eq19, SM2000
    
    # Update expected growth rate volatility
    eta[t] = eta[0] * np.exp(-kappa_eta * t) # Good med eq20, SM2000

    # Update revenue using stochastic process

    # R_real = R_real[:, t-1] * np.exp(
    #     (mu[:, t-1] - 0.5 * sigma[t-1]**2) * dt + sigma[t-1] * np.sqrt(dt) * Z1[:, t] # Good med eq17, SM2000
    # )

    R[:, t] = R[:, t-1] * np.exp(
        (mu[:, t-1] - lambda_1 * sigma[t-1] - 0.5 * sigma[t-1]**2) * dt + sigma[t-1] * np.sqrt(dt) * Z1[:, t] # Good med eq17, SM2000
    )

    



    # Compute costs and expenses
    COGS = alpha * R[:, t]
    other_expenses = F + beta * R[:, t]
    EBIT[:, t] = R[:, t] - COGS - other_expenses

    # Update cash balance
    X[:, t] = X[:, t-1] * np.exp(r_f * dt / 4) + EBIT[:, t] # Cash balance increases with EBIT
    
    # Only compute taxes yearly
    if t % 4 == 0:
        year_idx = t // 4
        taxable_income_yearly = np.maximum(np.sum(EBIT[:, t-3:t+1], axis=1) - L[:, year_idx-1], 0)
        taxes = taxrate * taxable_income_yearly
        L[:, year_idx] = np.maximum(L[:, year_idx-1] - np.sum(EBIT[:, t-3:t+1], axis=1), 0)  # Update yearly loss carryforward
        X[:, t] -= taxes  # Deduct yearly taxes from cash balance
    
    # Check for bankruptcy
    bankruptcy[active_firms, t] = X[active_firms, t] < 0  # Mark bankruptcy if cash is non-positive
    bankruptcy[bankruptcy[:, t], t:] = True  # Mark future time steps as bankrupt
    
    # If a company goes bankrupt, set all future values to zero
    # X[bankruptcy[:, t], t:] = 0
    # R[bankruptcy[:, t], t:] = 0
    # L[bankruptcy[:, t], t:] = 0
    # EBIT[bankruptcy[:, t], t:] = 0


# Compute Discounted Expected Value of the Firm
terminal_value = M * np.sum(EBIT[:, -4:], axis=1)  # Terminal value based on last year’s EBIT
# Adjust for bankrupt firms (ensures terminal value is also zero if bankrupt)
terminal_value[bankruptcy[:, -1]] = 0  # No terminal value if bankrupt
# Compute Expected Firm Value (DCF approach)
V0 = np.mean((X[:, -1] + terminal_value) * np.exp(-r_f * T))


# Print value estimate and number of bankruptcies
num_bankruptcies = np.sum(bankruptcy[:, -1])
print("\nEstimated Value (V0) using Discounted Free Cash Flow and Terminal Value:", V0)
print("Number of bankrupt simulations:", num_bankruptcies, "out of", simulations)


# Compute quantiles
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
revenue_quantiles = np.quantile(R, quantiles, axis=0)
cash_quantiles = np.quantile(X, quantiles, axis=0)
loss_quantiles = np.quantile(L, quantiles, axis=0)

# Compute quantiles
quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
revenue_quantiles = np.quantile(R[:, [3, 11, 19, 27, 39]], quantiles, axis=0)  # Extract values for 1, 3, 5, 7, 10 years forward

# Create a DataFrame
years_forward = [1, 3, 5, 7, 10]
percentile_labels = ["5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%", "55%", "60%", "65%", "70%", "75%", "80%", "85%", "90%", "95%", "Mean"]
revenue_table = pd.DataFrame(revenue_quantiles, columns=years_forward, index=percentile_labels[:-1])
revenue_table.loc["Mean"] = np.mean(R[:, [3, 11, 19, 27, 39]], axis=0)

# Print the table
print("\nTable 4. Revenue Distributions (millions)\n")
print(revenue_table)

