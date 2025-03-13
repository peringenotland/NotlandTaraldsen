import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Parameters
R_0 = 356  # Initial revenue in millions per quarter
L_0 = 559 # Loss-carryforward in millions
X_0 = 906 # Initial cash balance in millions
mu_0 = 0.11  # Initial growth rate per quarter
sigma_0 = 0.10  # Initial revenue volatility per quarter
eta_0 = 0.03 # Initial volatility of expected growth rate
rho = 0.0
mu_mean = 0.015  # Mean-reversion level for growth rate
sigma_mean = 0.05  # Mean-reversion level for volatility
taxrate = 0.35  # Corporate tax rate
r_f = 0.05  # Risk-free rate
kappa_mu = 0.07  # Mean-reversion speed for growth rate
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
num_steps = T * 4  # Convert years to quarters
revenues = np.zeros((simulations, num_steps))
cash_balances = np.zeros((simulations, num_steps))
mu_values = np.zeros((simulations, num_steps))
sigma_values = np.zeros((simulations, num_steps))
eta_values = np.zeros((simulations, num_steps))
loss_carryforward = np.zeros((simulations, num_steps))
bankruptcy = np.zeros((simulations, num_steps), dtype=bool)  # Track bankruptcy status



# Initial values
revenues[:, 0] = R_0
cash_balances[:, 0] = X_0
mu_values[:, 0] = mu_0
sigma_values[:, 0] = sigma_0
eta_values[:, 0] = eta_0
loss_carryforward[:, 0] = L_0
bankruptcy[:, 0] = False

np.random.seed(42)

# Monte Carlo simulation
for t in range(1, num_steps):

    # dW1 = np.random.normal(0, np.sqrt(dt), simulations) # Brownian motion for growth rate
    # dW2 = np.random.normal(0, np.sqrt(dt), simulations) # Brownian motion for volatility

    Z1 = np.random.normal(0, 1, simulations)  # Standard normal (ε_1)
    Z2 = np.random.normal(0, 1, simulations)  # Standard normal (ε_2)

    epsilon_1 = Z1
    epsilon_2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2  # Correlated noise

    dW1 = np.sqrt(dt) * epsilon_1
    dW2 = np.sqrt(dt) * epsilon_2




    # dW1 = np.random.normal(0, 1, simulations)
    # dW2 = np.random.normal(0, 1, simulations)
    
    # Update growth rate with mean-reversion
    mu_values[:, t] = np.exp(-kappa_mu * dt) * mu_values[:, t-1] + ((1 - np.exp(-kappa_mu * dt)) * (mu_mean - (lambda_2 * eta_values[:, t-1]) / kappa_mu)) + (np.sqrt((1 - np.exp(-2 * kappa_mu * dt)) / (2 * kappa_mu)) * eta_values[:, t-1] * dW2)
    
    # Update volatility with mean-reversion
    sigma_values[:, t] = sigma_0 * np.exp(-kappa_sigma * (t)) + sigma_mean * (1 - np.exp(-kappa_sigma * (t)))
    
    # Update expected growth rate volatility
    eta_values[:, t] = eta_0 * np.exp(-kappa_eta * (t))
    
    # Update revenue using stochastic process
    revenues[:, t] = revenues[:, t-1] * np.exp((mu_values[:, t-1] - lambda_1*sigma_values[:, t-1] - (0.5 * sigma_values[:, t-1]**2)) * dt + sigma_values[:, t-1] * dW1)
    
    # Compute costs and expenses
    COGS = alpha * revenues[:, t]
    other_expenses = F + beta * revenues[:, t]
    EBIT = revenues[:, t] - COGS - other_expenses

    # Update loss carry-forward
    loss_carryforward[:, t] = np.maximum(loss_carryforward[:, t-1] - EBIT * dt, 0)
    
    # Compute taxes
    taxable_income = np.maximum(EBIT - loss_carryforward[:, t], 0)
    taxes = taxrate * taxable_income
    
    # Update cash balance and check for bankruptcy
    cash_balances[:, t] = cash_balances[:, t-1] + EBIT - taxes
    bankruptcy[:, t] = cash_balances[:, t] < 0  # Mark bankruptcy if cash is non-positive
    
    # If a company goes bankrupt, set all future values to zero
    cash_balances[bankruptcy[:, t], t:] = 0
    revenues[bankruptcy[:, t], t:] = 0
    loss_carryforward[bankruptcy[:, t], t:] = 0

# Compute Discounted Cash Flow (DCF) valuation
V0 = np.mean((cash_balances[:, -1] + M * EBIT) * np.exp(-r_f * T))

# Print value estimate and number of bankruptcies
num_bankruptcies = np.sum(bankruptcy[:, -1])
print("\nEstimated Value (V0) using Discounted Free Cash Flow and Terminal Value:", V0)
print("Number of bankrupt simulations:", num_bankruptcies, "out of", simulations)



# # Compute expected values
# expected_revenue = np.mean(revenues, axis=0)
# expected_cash_balance = np.mean(cash_balances, axis=0)
# expected_loss_carryforward = np.mean(loss_carryforward, axis=0)

# # Print expected values
# print("Expected Revenue Over Time:", expected_revenue)
# print("Expected Cash Balance Over Time:", expected_cash_balance)
# print("Expected Loss Carry-Forward Over Time:", expected_loss_carryforward)

# # Plot a few revenue trajectories
# plt.figure(figsize=(10, 6))
# for i in range(10000):
#     plt.plot(revenues[i, :], alpha=0.3)
# plt.xlabel("Quarters")
# plt.ylabel("Revenue (millions)")
# plt.title("Monte Carlo Simulation of Revenue Over Time")
# plt.show()

# # Plot sigma
# plt.figure(figsize=(10, 6))
# for i in range(10000):
#     plt.plot(sigma_values[i, :], alpha=0.3)
# plt.xlabel("Quarters")
# plt.ylabel("Sigma")
# plt.title("Monte Carlo Simulation of Sigma Over Time")
# plt.show()

# # Plot mu
# plt.figure(figsize=(10, 6))
# for i in range(10000):
#     plt.plot(mu_values[i, :], alpha=0.3)
# plt.xlabel("Quarters")
# plt.ylabel("Mu")
# plt.title("Monte Carlo Simulation of Mu Over Time")
# plt.show()

# # Plot eta
# plt.figure(figsize=(10, 6))
# for i in range(10000):
#     plt.plot(eta_values[i, :], alpha=0.3)
# plt.xlabel("Quarters")
# plt.ylabel("Eta")
# plt.title("Monte Carlo Simulation of Eta Over Time")
# plt.show()

# # Plot cash balance trajectories
# plt.figure(figsize=(10, 6))
# for i in range(100):
#     plt.plot(cash_balances[i, :], alpha=0.3)
# plt.xlabel("Quarters")
# plt.ylabel("Cash Balance (millions)")
# plt.title("Monte Carlo Simulation of Cash Balance Over Time")
# plt.show()

# # Plot loss carry-forward trajectories
# plt.figure(figsize=(10, 6))
# for i in range(100):
#     plt.plot(loss_carryforward[i, :], alpha=0.3)
# plt.xlabel("Quarters")
# plt.ylabel("Loss Carry-Forward (millions)")
# plt.title("Monte Carlo Simulation of Loss Carry-Forward Over Time")
# plt.show()

# Compute quantiles
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
revenue_quantiles = np.quantile(revenues, quantiles, axis=0)
cash_quantiles = np.quantile(cash_balances, quantiles, axis=0)
loss_quantiles = np.quantile(loss_carryforward, quantiles, axis=0)

# # Plot revenue quantiles
# plt.figure(figsize=(10, 6))
# for q, label in zip(revenue_quantiles, ['5%', '25%', '50%', '75%', '95%']):
#     plt.plot(q, label=f'{label} Quantile')
# plt.xlabel("Quarters")
# plt.ylabel("Revenue (millions)")
# plt.title("Revenue Quantiles Over Time")
# plt.legend()
# plt.show()

# Plot cash balance quantiles
# plt.figure(figsize=(10, 6))
# for q, label in zip(cash_quantiles, ['5%', '25%', '50%', '75%', '95%']):
#     plt.plot(q, label=f'{label} Quantile')
# plt.xlabel("Quarters")
# plt.ylabel("Cash Balance (millions)")
# plt.title("Cash Balance Quantiles Over Time")
# plt.legend()
# plt.show()

# # Plot loss carry-forward quantiles
# plt.figure(figsize=(10, 6))
# for q, label in zip(loss_quantiles, ['5%', '25%', '50%', '75%', '95%']):
#     plt.plot(q, label=f'{label} Quantile')
# plt.xlabel("Quarters")
# plt.ylabel("Loss Carry-Forward (millions)")
# plt.title("Loss Carry-Forward Quantiles Over Time")
# plt.legend()
# plt.show()

# Compute quantiles
quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
revenue_quantiles = np.quantile(revenues[:, [4, 12, 20, 28, 40]], quantiles, axis=0)  # Extract values for 1, 3, 5, 7, 10 years forward

# Create a DataFrame
years_forward = [1, 3, 5, 7, 10]
percentile_labels = ["5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%", "55%", "60%", "65%", "70%", "75%", "80%", "85%", "90%", "95%", "Mean"]
revenue_table = pd.DataFrame(revenue_quantiles, columns=years_forward, index=percentile_labels[:-1])
revenue_table.loc["Mean"] = np.mean(revenues[:, [4, 12, 20, 28, 40]], axis=0)

# Print the table
print("\nTable 4. Revenue Distributions (millions)\n")
print(revenue_table)

# # Plot revenue quantiles
# plt.figure(figsize=(10, 6))
# for q, label in zip(revenue_quantiles, percentile_labels[:-1]):
#     plt.plot(years_forward, q, label=f'{label} Quantile')
# plt.xlabel("Years Forward")
# plt.ylabel("Revenue (millions)")
# plt.title("Revenue Quantiles Over Time")
# plt.legend()
# plt.show()


