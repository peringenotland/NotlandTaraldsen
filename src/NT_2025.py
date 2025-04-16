import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import parameters as p


# Parameters
gvkey = 232646

R_0 = p.get_R_0(gvkey)  # Initial revenue in millions per quarter
L_0 = p.get_L_0(gvkey) # Loss-carryforward in millions
X_0 = p.get_X_0(gvkey) # Initial cash balance in millions
CapEx_Ratio_0 = p.get_Initial_CapEx_Ratio(gvkey)
CapEx_Ratio_longterm = p.get_Long_term_CAPEX(gvkey)
Dep_Ratio = p.get_Depreciation_Ratio(gvkey)
PPE_0 = p.get_PPE_0(gvkey)
mu_0 = p.get_mu_0()  # Initial growth rate per quarter
sigma_0 = p.get_sigma_0()  # Initial revenue volatility per quarter
# sigma_0 = 0.17
eta_0 = p.get_eta_0(gvkey) # Initial volatility of expected growth rate
rho = p.get_rho() # Correlation between revenue and growth rate
mu_mean = p.get_mu_mean()  # Mean-reversion level for growth rate
sigma_mean = p.get_sigma_mean()  # Mean-reversion level for volatility
taxrate = p.get_taxrate(gvkey)  # Corporate tax rate
r_f = p.get_r_f()  # Risk-free rate
kappa_mu = p.get_kappa_mu()  # Mean-reversion speed for expected growth rate
kappa_mu = 0.09
kappa_sigma = p.get_kappa_sigma()  # Mean-reversion speed for volatility
kappa_eta = p.get_kappa_eta()  # Mean-reversion speed for expected growth rate volatility
kappa_gamma = p.get_kappa_gamma()
kappa_phi = p.get_kappa_phi()
kappa_capex = p.get_kappa_capex()
gamma_0 = p.get_gamma_0(gvkey)
gamma_mean = p.get_gamma_mean(gvkey)
phi_0 = p.get_phi_0()
phi_mean = p.get_phi_mean()
lambda_R = p.get_lambda_R() # Market price of risk for the revenue factor
lambda_mu = p.get_lambda_mu() # Market price of risk for the expected rate of growth in revenues factor
lambda_gamma = p.get_lambda_gamma()
T = p.get_T()  # Time horizon in years
dt = p.get_dt() # Time step
M = p.get_M() # Exit multiple
simulations = p.get_simulations()  # Number of Monte Carlo runs


# Simulation setup
num_steps = T * 4 + 1 # Quarters in 25 years + initial step
np.random.seed(42) # Seed random number generator

R = np.zeros((simulations, num_steps)) # Revenue trajectories
X = np.zeros((simulations, num_steps)) # Cash balance trajectories
Cost = np.zeros((simulations, num_steps)) # Cost
CapEx = np.zeros((simulations, num_steps))
CapEx_ratio = np.zeros(num_steps)
Dep = np.zeros((simulations, num_steps))
PPE = np.zeros((simulations, num_steps))
Tax = np.zeros((simulations, num_steps))
mu = np.zeros((simulations, num_steps)) # Growth rate trajectories
NOPAT = np.zeros((simulations, num_steps)) # Net Operating Profit after Tax
gamma = np.zeros((simulations, num_steps))
phi = np.zeros(num_steps)
sigma = np.zeros(num_steps)
eta = np.zeros(num_steps)
L = np.zeros((simulations, num_steps)) # Loss carry-forward trajectories
# L = np.zeros((simulations, T+1), dtype=np.float32)  # Loss carryforward tracked yearly
bankruptcy = np.zeros((simulations, num_steps), dtype=bool) # Bankruptcy indicator
EBITDA = np.zeros((simulations, num_steps))  # Earnings before interest and taxes


# Initial values (t = 0) not simulated
R[:, 0] = R_0 # Initial revenue
X[:, 0] = X_0 # Initial cash balance
Cost[:, 0] = gamma_0 * R_0
CapEx[:, 0] = CapEx_Ratio_0 * R_0
CapEx_ratio[0] = CapEx_Ratio_0
PPE[:, 0] = PPE_0
Dep[:, 0] = np.nan
Tax[:, 0] = np.nan
NOPAT[:, 0] = np.nan
mu[:, 0] = mu_0 # Initial growth rate
gamma[:, 0] = gamma_0
phi[0] = phi_0
sigma[0] = sigma_0
eta[0] = eta_0
L[:, 0] = L_0 # Initial loss carry-forward
bankruptcy[:, 0] = False # Initial bankruptcy indicator

# Generate correlated random shocks
Z_R = np.random.randn(simulations, num_steps)  # Standard normal noise for revenue
Z_mu = rho * Z_R + np.sqrt(1 - rho**2) * np.random.randn(simulations, num_steps)  # Correlated noise for growth
Z_gamma = np.random.randn(simulations, num_steps)


# Monte Carlo simulation
for t in range(1, num_steps):

    # Only update non-bankrupt firms
    active_firms = ~bankruptcy[:, t-1]  # Firms that haven't gone bankrupt
    
    # Update revenue using stochastic process
    R[:, t] = R[:, t-1] * np.exp(
            (mu[:, t-1] - lambda_R * sigma[t-1] - 0.5 * sigma[t-1]**2) * dt + sigma[t-1] * np.sqrt(dt) * Z_R[:, t] # Good med eq28, SchosserStröbele
        )
    
    # Update growth rate with mean-reversion
    mu[:, t] = np.exp(-kappa_mu * dt) * mu[:, t-1] + (1 - np.exp(-kappa_mu * dt)) * (mu_mean - ((lambda_mu*eta[t-1])/(kappa_mu))) + np.sqrt((1 - np.exp(-2*kappa_mu*dt))/(2*kappa_mu)) * eta[t-1] * Z_mu[:, t] # Good med eq29 i SchosserStröbele

    # Gamma (cost ratio to revenue)
    gamma[:, t] = np.exp(-kappa_gamma*dt) * gamma[:, t-1] + (1 - np.exp(-kappa_gamma*dt)) * (gamma_mean - ((lambda_gamma * phi[t-1])/(kappa_gamma))) + np.sqrt((1 - np.exp(-2*kappa_gamma*dt))/(2*kappa_gamma)) * phi[t-1] * Z_gamma[:, t] 


    # Sigma (volatility in revenue)
    sigma[t] = sigma[0] * np.exp(-kappa_sigma * t) + sigma_mean * (1 - np.exp(-kappa_sigma * t)) # Good med eq19 SM2000 og eq32 SchosserStröbele
    
    # Update expected growth rate volatility
    eta[t] = eta[0] * np.exp(-kappa_eta * t) # Good med eq20, SM2000 og schosser strobele eq33

    # Phi
    phi[t] = np.exp(-kappa_phi * t) * phi[0] + (1 - np.exp(-kappa_phi * t)) * phi_mean # Eq 34 in SchosserStrobele and eq30 in SM2001

    # CapEx ratio 
    CapEx_ratio[t] = CapEx_ratio[0] * np.exp(-kappa_capex * t) + CapEx_Ratio_longterm * (1 - np.exp(-kappa_capex * t)) # Good med eq19 SM2000 og eq32 SchosserStröbele

    # Cost
    Cost[:, t] = gamma[:, t] * R[:, t] # Vi bruker total cost ratio, gamma er excluding depreciation og amortizzzation, but including interest expense.

    # Depreciation
    Dep[:, t] = Dep_Ratio * PPE[:, t-1]

    # CapEx # TODO: Se på Capex process, nå er det kun initial ratio * revenue

    CapEx[:, t] = CapEx_ratio[t] * R[:, t]

    # PPE
    PPE[:, t] = PPE[:, t-1] - Dep[:, t] + CapEx[:, t] 

    # Compute Tax in absolute value (eq14 in SchosserStrobele)
    # Tax is computed quarterly, and determined using loss-carryforward from company data. 
    # TODO: Discuss in thesis!
    # if (R[:, t] - Cost[:, t] - Dep[:, t] - L[:, t-1]) <= 0:
    #     Tax[:, t] = 0
    # else:
    #     Tax[:, t] = (R[:, t] - Cost[:, t] - Dep[:, t] - L[:, t-1]) * taxrate
    # Det over angående tax ga feilmelding så endret til det under:
    taxable_income = R[:, t] - Cost[:, t] - Dep[:, t] - L[:, t-1]
    Tax[:, t] = np.where(taxable_income <= 0, 0, taxable_income * taxrate)


    NOPAT[:, t] = R[:, t] - Cost[:, t] - Dep[:, t] - Tax[:, t]

    # Compute Loss Carryforward
    # if L[:, t-1] > (NOPAT[:, t] + Tax[:, t]):
    #     L[:, t] = L[:, t-1] - (NOPAT[:, t] + Tax[:, t])
    # else:
    #     L[:, t] = 0
    # Det over angående loss carryforward ga feilmelding pga if og else på en vektor.
    used_loss = NOPAT[:, t] + Tax[:, t]
    L[:, t] = np.where(
        L[:, t-1] > used_loss,
        L[:, t-1] - used_loss,
        0
    )

    # Update cash balance
    X[:, t] = X[:, t-1] + (r_f * X[:, t-1] + NOPAT[:, t] + Dep[:, t] - CapEx[:, t]) * dt
    
    ## TODO på onsdag:
    # Check for bankruptcy
    bankruptcy[active_firms, t] = X[active_firms, t] < 0  # Mark bankruptcy if cash is non-positive
    bankruptcy[bankruptcy[:, t], t:] = True  # Mark future time steps as bankrupt
    
    # If a company goes bankrupt, set all future values to zero
    X[bankruptcy[:, t], t:] = 0
    R[bankruptcy[:, t], t:] = 0
    L[bankruptcy[:, t], t:] = 0
    EBITDA[bankruptcy[:, t], t:] = 0


# Compute Discounted Expected Value of the Firm #### EBITDA
terminal_value = M * (R[:, -1] - Cost[:, -1])
# Adjust for bankrupt firms (ensures terminal value is also zero if bankrupt)
terminal_value[bankruptcy[:, -1]] = 0  # No terminal value if bankrupt
# Compute Expected Firm Value (DCF approach)
V0 = np.mean((X[:, -1] + terminal_value) * np.exp(-r_f * T))


# Print value estimate and number of bankruptcies
num_bankruptcies = np.sum(bankruptcy[:, -1])
print("\nEstimated Value (V0) using Discounted Free Cash Flow and Terminal Value:", V0)
print("Number of bankrupt simulations:", num_bankruptcies, "out of", simulations)



# Compute quantiles
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95, 0.995]
revenue_quantiles = np.quantile(R, quantiles, axis=0)
cash_quantiles = np.quantile(X, quantiles, axis=0)
loss_quantiles = np.quantile(L, quantiles, axis=0)

# Compute quantiles
quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.995]
revenue_quantiles = np.quantile(R[:, [3, 11, 19, 27, 39, 59, 79, 99]], quantiles, axis=0)  # Extract values for 1, 3, 5, 7, 10 years forward

# Create a DataFrame
years_forward = [1, 3, 5, 7, 10, 15, 20, 25]
percentile_labels = ["5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%", "55%", "60%", "65%", "70%", "75%", "80%", "85%", "90%", "95%", "99.5%", "Mean"]
revenue_table = pd.DataFrame(revenue_quantiles, columns=years_forward, index=percentile_labels[:-1])
revenue_table.loc["Mean"] = np.mean(R[:, [3, 11, 19, 27, 39, 59, 79, 99]], axis=0)

plt.figure(figsize=(12, 6))

# Plot revenue
for i in range(200):  # Ikke mer enn 200, det holder for å se variasjon
    plt.plot(R[i, :], alpha=0.05, color='blue')
    
# Plot cash balance
for i in range(200):
    plt.plot(X[i, :], alpha=0.05, color='green')

# Legg til gjennomsnitt
plt.plot(np.mean(R, axis=0), color='blue', linewidth=2, label="Mean Revenue")
plt.plot(np.mean(X, axis=0), color='green', linewidth=2, label="Mean Cash Balance")

plt.xlabel("Quarters")
plt.ylabel("Millions")
plt.title("Monte Carlo Simulation: Revenue and Cash Over Time")
plt.legend()
plt.grid(True)
plt.show()


# Print the table
print("\nTable 4. Revenue Distributions (millions)\n")
print(revenue_table)

# Samle alle parametere i en liste av dicts
parameter_data = [
    {"Parameter": "R_0", "Value": R_0, "Description": "Initial revenue (millions/quarter)"},
    {"Parameter": "L_0", "Value": L_0, "Description": "Initial loss carry-forward (millions)"},
    {"Parameter": "X_0", "Value": X_0, "Description": "Initial cash balance (millions)"},
    {"Parameter": "CapEx_Ratio_0", "Value": CapEx_Ratio_0, "Description": "Initial CapEx-to-revenue ratio"},
    {"Parameter": "CapEx_Ratio_longterm", "Value": CapEx_Ratio_longterm, "Description": "Long-term CapEx-to-revenue ratio"},
    {"Parameter": "Dep_Ratio", "Value": Dep_Ratio, "Description": "Depreciation-to-PPE ratio"},
    {"Parameter": "PPE_0", "Value": PPE_0, "Description": "Initial PPE (Property, Plant, Equipment)"},
    {"Parameter": "mu_0", "Value": mu_0, "Description": "Initial revenue growth rate"},
    {"Parameter": "mu_mean", "Value": mu_mean, "Description": "Mean reversion level for growth rate"},
    {"Parameter": "kappa_mu", "Value": kappa_mu, "Description": "Mean reversion speed for growth rate"},
    {"Parameter": "sigma_0", "Value": sigma_0, "Description": "Initial revenue volatility"},
    {"Parameter": "sigma_mean", "Value": sigma_mean, "Description": "Long-term revenue volatility"},
    {"Parameter": "kappa_sigma", "Value": kappa_sigma, "Description": "Mean reversion speed for volatility"},
    {"Parameter": "eta_0", "Value": eta_0, "Description": "Initial growth rate volatility"},
    {"Parameter": "kappa_eta", "Value": kappa_eta, "Description": "Mean reversion speed for eta"},
    {"Parameter": "rho", "Value": rho, "Description": "Correlation between revenue and growth rate"},
    {"Parameter": "gamma_0", "Value": gamma_0, "Description": "Initial cost-to-revenue ratio"},
    {"Parameter": "gamma_mean", "Value": gamma_mean, "Description": "Long-term cost-to-revenue ratio"},
    {"Parameter": "kappa_gamma", "Value": kappa_gamma, "Description": "Mean reversion speed for gamma"},
    {"Parameter": "phi_0", "Value": phi_0, "Description": "Initial volatility of cost ratio"},
    {"Parameter": "phi_mean", "Value": phi_mean, "Description": "Long-term volatility of cost ratio"},
    {"Parameter": "kappa_phi", "Value": kappa_phi, "Description": "Mean reversion speed for phi"},
    {"Parameter": "lambda_R", "Value": lambda_R, "Description": "Market price of risk (revenue)"},
    {"Parameter": "lambda_mu", "Value": lambda_mu, "Description": "Market price of risk (growth rate)"},
    {"Parameter": "lambda_gamma", "Value": lambda_gamma, "Description": "Market price of risk (cost ratio)"},
    {"Parameter": "taxrate", "Value": taxrate, "Description": "Corporate tax rate"},
    {"Parameter": "r_f", "Value": r_f, "Description": "Risk-free rate"},
    {"Parameter": "T", "Value": T, "Description": "Time horizon (years)"},
    {"Parameter": "dt", "Value": dt, "Description": "Time step"},
    {"Parameter": "M", "Value": M, "Description": "Exit multiple (terminal value)"},
    {"Parameter": "simulations", "Value": simulations, "Description": "Number of Monte Carlo simulations"}
]

# Lag tabellen
param_table = pd.DataFrame(parameter_data)

# Skriv ut
print("\nTable X. Model Parameters\n")
print(param_table.to_string(index=False))

# Beregn verdi for hver simulering
firm_value = (X[:, -1] + terminal_value) * np.exp(-r_f * T)

# Filtrer bort topp 5% (behold de laveste 95%)
cutoff_95 = np.percentile(firm_value, 95)
filtered_values = firm_value[firm_value <= cutoff_95]

# Lag histogram
plt.figure(figsize=(10, 6))
plt.hist(filtered_values, bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='grey', label='Bottom 95%')

# Legg til vertikale linjer
mean_val = np.mean(filtered_values)
median_val = np.median(filtered_values)

plt.axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean (Bottom 95%): {mean_val:.1f}')
plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median (Bottom 95%): {median_val:.1f}')

plt.title("Distribution of Simulated Company Valuations (Bottom 95%)")
plt.xlabel("Present Value (millions)")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
