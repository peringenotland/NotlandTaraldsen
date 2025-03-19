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
rho = 0.0  # Correlation between revenue and growth shocks
mu_mean = 0.015  # Long-term mean growth rate
sigma_mean = 0.05  # Long-term volatility mean

taxrate = 0.35  # Corporate tax rate
r_f = 0.05  # Risk-free rate

kappa_mu = 0.07  # Growth rate mean reversion speed
kappa_sigma = 0.07  # Volatility mean reversion speed
kappa_eta = 0.07  # Expected growth rate volatility mean reversion speed

alpha = 0.75 # COGS as a percentage of revenues
F = 75 # Fixed costs in millions per quarter
beta = 0.19 # Variable component of other expenses

lambda_1 = 0.01 # Market price of risk for revenue factor
lambda_2 = 0.0 # Market price of risk for expected revenue growth factor

T = 25  # Time horizon in years
dt = 1  # Time step (quarterly)
M = 10 # Exit multiple
simulations = 100000  # Reduced number of Monte Carlo runs to reduce memory usage
num_steps = int(T * 4)  # Convert years to quarters
np.random.seed(42)


# Initialize arrays
R = np.zeros((simulations, num_steps), dtype=np.float32)
L = np.zeros((simulations, num_steps), dtype=np.float32)
X = np.zeros((simulations, num_steps), dtype=np.float32)
mu = np.zeros((simulations, num_steps), dtype=np.float32)
sigma = np.zeros(num_steps, dtype=np.float32)
eta = np.zeros(num_steps, dtype=np.float32)
bankruptcy = np.zeros((simulations, num_steps), dtype=bool)


# Set initial values (t = 0)
R[:, 0] = R_0
L[:, 0] = L_0
X[:, 0] = X_0
mu[:, 0] = mu_0
sigma[0] = sigma_0
eta[0] = eta_0
bankruptcy[:, 0] = False

# Generate correlated random shocks for revenue and growth rate
Z1 = np.random.randn(simulations, num_steps).astype(np.float32)  # Standard normal noise for revenue fluctuations
Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(simulations, num_steps).astype(np.float32)  # Correlated growth rate noise

# Monte Carlo Simulation
for t in range(1, num_steps):

    sigma[t] = sigma[0] * np.exp(-kappa_sigma * t) + sigma_mean * (1 - np.exp(-kappa_sigma * t)) # Good med eq19, SM2000
    eta[t] = eta[0] * np.exp(-kappa_eta * t) # Good med eq20, SM2000

    # Update revenues using geometric Brownian motion
    R[:, t] = R[:, t-1] * np.exp(
        (mu[:, t-1] - lambda_1 * sigma[t-1] - 0.5 * sigma[t-1]**2) * dt + sigma[t-1] * np.sqrt(dt) * Z1[:, t] # Good med eq17, SM2000
    )
    
    # Update growth rate with mean reversion
    mu[:, t] = np.exp(-kappa_mu * dt) * mu[:, t-1] + (1 - np.exp(-kappa_mu * dt)) * (mu_mean - (lambda_2*eta[t-1])/(kappa_mu)) + np.sqrt((1 - np.exp(-2*kappa_mu*dt))/(2*kappa_mu)) * eta[t-1] * np.sqrt(dt) * Z2[:, t] # Good med eq18, SM2000
    
    # Calculate costs
    COGS = alpha * R[:, t]  # Cost of Goods Sold
    other_expenses = beta * R[:, t] + F  # Other expenses including fixed costs
    EBIT = R[:, t] - COGS - other_expenses  # Earnings before interest and taxes
    
    # Tax calculation
    taxable_income = np.maximum(EBIT - L[:, t-1], 0)  # Only positive taxable income is taxed
    taxes = taxrate * taxable_income
    L[:, t] = np.maximum(L[:, t-1] - EBIT, 0)  # Update loss carryforward
    
    # Net Income calculation
    net_income = EBIT - taxes
    
    # Update cash balance
    X[:, t] = X[:, t-1] + net_income * dt  # Cash balance increases with net income
    
    # Check for bankruptcy (if cash balance goes negative)
    bankruptcy[:, t] = X[:, t] < 0
    
    # If bankrupt, set future cash balances to zero
    X[bankruptcy[:, t], t:] = 0
    R[bankruptcy[:, t], t:] = 0
    L[bankruptcy[:, t], t:] = 0
    mu[bankruptcy[:, t], t:] = 0

# Compute expected firm value at final time step using exit multiple approach
exit_value = M * EBIT
firm_values = np.where(bankruptcy[:, -1], 0, exit_value + X[:, -1] * np.exp(-r_f * T))  # If bankrupt, firm value is zero

# Compute expected firm value
expected_firm_value = np.mean(firm_values)

# Display results
print(f"Expected Firm Value (Present Value): {expected_firm_value:.2f} million")

# Plot revenue distribution at final time step
plt.figure(figsize=(10, 5))
plt.hist(R[:, -1], bins=50, alpha=0.75, edgecolor='black')
plt.xlabel("Final Revenue (Million)")
plt.ylabel("Frequency")
plt.title("Distribution of Final Revenue Across Simulations")
plt.show()

# Convert simulation results to DataFrame and display
results_df = pd.DataFrame({
    "Final Revenue": R[:, -1],
    "Final Cash Balance": X[:, -1],
    "Bankruptcy": bankruptcy[:, -1],
    "Firm Value": firm_values
})

print(results_df)