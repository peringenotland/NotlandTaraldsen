import numpy as np
import matplotlib.pyplot as plt

def simulate_firm_value(
    S0, mu0, sigma0, c0, k_mu, k_sigma, k_c, sigma_c, T, dt, r, simulations
):
    """
    Monte Carlo simulation for firm valuation using Schwartz-Moon model.
    
    Parameters:
        S0 (float): Initial revenue.
        mu0 (float): Initial expected revenue growth rate.
        sigma0 (float): Initial revenue volatility.
        c0 (float): Initial variable cost ratio.
        k_mu (float): Mean-reversion speed for revenue growth.
        k_sigma (float): Mean-reversion speed for revenue volatility.
        k_c (float): Mean-reversion speed for cost ratio.
        sigma_c (float): Volatility of cost ratio.
        T (float): Time horizon in years.
        dt (float): Time step.
        r (float): Risk-free rate.
        simulations (int): Number of Monte Carlo simulations.
    
    Returns:
        np.ndarray: Simulated firm values at time T.
    """
    steps = int(T / dt)
    firm_values = np.zeros(simulations)
    
    for i in range(simulations):
        S, mu, sigma, c = S0, mu0, sigma0, c0
        for _ in range(steps):
            dW1, dW2, dW3 = np.random.normal(0, np.sqrt(dt), 3)
            mu += k_mu * (0.03 - mu) * dt + sigma * dW1  # Mean-reverting growth
            sigma += k_sigma * (0.05 - sigma) * dt + sigma * dW2  # Mean-reverting volatility
            c += k_c * (0.8 - c) * dt + sigma_c * dW3  # Mean-reverting cost ratio
            S *= np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW1)
        EBITDA = S * (1 - c)
        firm_values[i] = EBITDA * 10  # Terminal value using multiple
    
    return firm_values

# Parameters
S0 = 100  # Initial revenue
mu0 = 0.1  # Initial growth rate
sigma0 = 0.2  # Initial revenue volatility
c0 = 0.7  # Initial variable cost ratio
k_mu = 0.1  # Mean-reversion speed for growth rate
k_sigma = 0.1  # Mean-reversion speed for volatility
k_c = 0.2  # Mean-reversion speed for cost ratio
sigma_c = 0.05  # Volatility of cost ratio
T = 10  # Time horizon in years
dt = 0.1  # Time step
r = 0.05  # Risk-free rate
simulations = 10000  # Number of Monte Carlo runs

# Run simulation
firm_values = simulate_firm_value(S0, mu0, sigma0, c0, k_mu, k_sigma, k_c, sigma_c, T, dt, r, simulations)

# Plot results
plt.hist(firm_values, bins=50, alpha=0.7, color='b', edgecolor='black')
plt.xlabel("Firm Value")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulation of Firm Valuation (Schwartz-Moon Model)")
plt.grid()
plt.show()

# Print summary statistics
print(f"Mean Firm Value: {np.mean(firm_values):.2f}")
print(f"Median Firm Value: {np.median(firm_values):.2f}")
print(f"95% Confidence Interval: [{np.percentile(firm_values, 2.5):.2f}, {np.percentile(firm_values, 97.5):.2f}]")
