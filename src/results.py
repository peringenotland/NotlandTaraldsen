# ------------------------------------------------------------
# results.py
# ------------------------------------------------------------
# This script is used to present the results from a saved model.
# The scripts contains different methods for plotting and printing 
# results.
# ------------------------------------------------------------
# Version 3, LongstaffSchwartz inspired Financing. 
# -> Optimal Control problem with dynamic financing decision.
# Version 3, Gamba Abandonment value for bankruptcy handling.
#
# Authors: 
# Per Inge Notland
# David Taraldsen
# 
# Date: 25.04.2025
# ------------------------------------------------------------


import pickle
import parameters_v4 as p
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_latest_simulation_results(gvkey, version=3):
    """
    Load simulation results from a pickle file.

    Parameters:
    - gvkey (str): The gvkey of the firm.

    Returns:
    - dict: A dictionary containing the simulation results.
    """
    # Path to the file you saved earlier
    filepath = f"simulation_outputs_latest/v{version}_{gvkey}_latest_sim_results.pkl"

    # Load the file
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    # Now `data` is a dictionary with two top-level keys: "parameters" and "results"
    return data


def print_main_results(firm_data):
    '''
    Print the results of the simulation.
    '''
    print("\nMonte Carlo Simulation Results")
    print("===================================")
    print("GVKEY:", firm_data["gvkey"])
    print("Name:", firm_data["name"])
    print("Timestamp: ", firm_data["timestamp"])
    V0 = firm_data["results"]["V0"]
    print("\nEstimated Value (V0): ", V0.round(2), "millions")
    num_bankruptcies = firm_data["results"]["num_bankruptcies"]
    simulations = firm_data["parameters"]["simulations"]
    print("Bankrupt simulations:", num_bankruptcies, "out of", simulations, ",   ", (num_bankruptcies/simulations)*100, "%")


def print_all_parameters(firm_data):
    '''
    Print all parameters.
    '''
    print("\nModel Parameters")
    print("===================================")
    for key, value in firm_data["parameters"].items():
        print(f"{key}: {value}")


def plot_revenue_and_cash(data):
    plt.figure(figsize=(12, 6))
    R = data["results"]["R"]
    X = data["results"]["X"]

    # # Plot revenue
    # for i in range(200):  # Ikke mer enn 200, det holder for å se variasjon
    #     plt.plot(R[i, :], alpha=0.05, color='blue')
        
    # # Plot cash balance
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

def print_revenue_distributions(data):
    R = data["results"]["R"]
    # Compute quantiles
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95, 0.995]
    revenue_quantiles = np.quantile(R, quantiles, axis=0)

    # Compute quantiles
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.995]
    revenue_quantiles = np.quantile(R[:, [3, 11, 19, 27, 39, 59, 79, 99]], quantiles, axis=0)  # Extract values for 1, 3, 5, 7, 10 years forward

    # Create a DataFrame
    years_forward = [1, 3, 5, 7, 10, 15, 20, 25]
    percentile_labels = ["5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%", "55%", "60%", "65%", "70%", "75%", "80%", "85%", "90%", "95%", "99.5%", "Mean"]
    revenue_table = pd.DataFrame(revenue_quantiles, columns=years_forward, index=percentile_labels[:-1])
    revenue_table.loc["Mean"] = np.mean(R[:, [3, 11, 19, 27, 39, 59, 79, 99]], axis=0)

    # Print the table
    print("\nTable 4. Revenue Distributions (millions)\n")
    print(revenue_table)


def plot_firm_value_distribution(data):
    '''
    Plot the distribution of firm values from the simulation,
    showing the middle 95% of values (excluding top 2.5% and bottom 2.5%).
    '''
    # Retrieve simulation data
    r_f = data["parameters"]["r_f"]
    T = data["parameters"]["T"]
    terminal_value = data["results"]["terminal_value"]

    # Compute firm value for each simulation
    firm_value = terminal_value * np.exp(-r_f * T)

    # Calculate mid 95% (remove top and bottom 2.5%)
    lower_bound = np.percentile(firm_value, 2.5)
    upper_bound = np.percentile(firm_value, 97.5)
    mid_95_values = firm_value[(firm_value >= lower_bound) & (firm_value <= upper_bound)]

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(mid_95_values, bins=100, density=True, alpha=0.7,
             color='skyblue', edgecolor='grey', label='Middle 95%')

    # Vertical lines for mean and median
    mean_val_all = np.mean(firm_value)
    median_val_all = np.median(firm_value)

    plt.axvline(mean_val_all, color='blue', linestyle='--', linewidth=2,
                label=f'Mean (All): {mean_val_all:.1f}')
    plt.axvline(median_val_all, color='green', linestyle='--', linewidth=2,
                label=f'Median (All): {median_val_all:.1f}')

    plt.title("Distribution of Simulated Terminal Values Discounted before financing (Middle 95%)")
    plt.xlabel("Present Value (millions)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_revenue_trajectories(data, num_trajectories=100):
    '''
    Plot a few revenue trajectories from the simulation.
    '''
    # Hent data fra simuleringen
    R = data["results"]["R"]

    # Plot a few revenue trajectories with the mean
    plt.figure(figsize=(10, 6))

    # Plot all individual simulations with low opacity
    for i in range(num_trajectories):
        plt.plot(R[i, :], alpha=0.3)

    # Plot the mean trajectory
    mean_trajectory = R.mean(axis=0)
    plt.plot(mean_trajectory, color='black', linewidth=2.5, label='Mean Trajectory')

    # Labels and title
    plt.xlabel("Quarters")
    plt.ylabel("Revenue (millions)")
    plt.title("Monte Carlo Simulation of Revenue Over Time")
    plt.legend()
    plt.show()


def plot_bankruptcy_timeline(data):
    """
    Plot when bankruptcies occur over time (new bankruptcies only).

    Shows the number of new bankruptcies occurring in each quarter.
    """
    bankruptcy = data["results"]["bankruptcy"]  # shape (simulations, num_steps)

    # Find new bankruptcies per time step: 1 only where it transitions from 0 → 1
    new_bankruptcies = np.diff(bankruptcy.astype(int), axis=1)
    new_bankruptcies = np.maximum(new_bankruptcies, 0)  # Only keep positive transitions

    # Sum across simulations to get count per time step
    bankruptcies_per_step = np.sum(new_bankruptcies, axis=0)

    # Plot
    time = np.arange(1, bankruptcy.shape[1])  # diff reduces length by 1
    plt.figure(figsize=(10, 5))
    plt.bar(time, bankruptcies_per_step, color='salmon', alpha=0.85)
    plt.xlabel("Quarter")
    plt.ylabel("New Bankruptcies")
    plt.title("New Bankruptcies Per Quarter")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_bankruptcy_matrix(data, max_rows=20):
    """
    Print the bankruptcy matrix showing which simulations went bankrupt and when.
    
    Parameters:
    - data (dict): Loaded simulation data
    - max_rows (int): Limit the number of simulation rows shown (default=20)
    """
    bankruptcy = data["results"]["bankruptcy"]
    df = pd.DataFrame(bankruptcy.astype(int))  # 1 = bankruptcy, 0 = not

    df.index.name = "Simulation"
    df.columns = [f"Q{t}" for t in range(df.shape[1])]

    print("\nBankruptcy Matrix (1 = bankrupt)")
    if df.shape[0] > max_rows:
        print(df.head(max_rows).to_string())
        print(f"\n[Only first {max_rows} of {df.shape[0]} simulations shown]")
    else:
        print(df.to_string())

def plot_financing(data):
    """
    Plot the amount of financing raised over time.

    Parameters:
    - data: Dictionary loaded from the simulation results (via get_latest_simulation_results)
    """
    financing = data["results"].get("financing")
    if financing is None:
        print("No financing data found in results.")
        return

    num_steps = financing.shape[1]
    time = np.arange(num_steps)

    total_raised = np.sum(financing, axis=0)
    num_financed = np.sum(financing > 0, axis=0)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.bar(time, total_raised, color='dodgerblue', alpha=0.7, label='Total Financing Raised (M)')
    ax1.set_xlabel("Quarter")
    ax1.set_ylabel("Financing Raised (Millions)", color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.grid(True)

    # Twin axis for number of firms raising capital
    ax2 = ax1.twinx()
    ax2.plot(time, num_financed, color='darkred', linestyle='--', linewidth=2, label='Number of Firms Financed')
    ax2.set_ylabel("Number of Simulations Receiving Financing", color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')

    plt.title("Financing Activity Over Time")
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

def plot_betas(data):
    betas = data["results"].get("betas")
    if betas is None:
        print("No beta regression data found.")
        return
    labels = ["Cash (β₁)", "Revenue (β₂)", "Cash² (β₃)"]
    plt.figure(figsize=(12, 6))
    for i in range(1, betas.shape[1]):  # Skip intercept
        plt.plot(betas[:, i], label=labels[i - 1])
    plt.xlabel("Quarter")
    plt.ylabel("Beta Coefficient")
    plt.title("Evolution of Regression Coefficients (Betas) Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_bankruptcy_rate_over_time(data):
    bankruptcy = data["results"]["bankruptcy"]
    bankruptcy_rate = np.mean(bankruptcy, axis=0)  # Average bankruptcy rate per timestep
    time_steps = np.arange(bankruptcy.shape[1])

    plt.figure(figsize=(10,6))
    plt.plot(time_steps, bankruptcy_rate * 100)
    plt.title("Bankruptcy Rate Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Percentage of bankrupt firms (%)")
    plt.grid(True)
    plt.show()

def plot_financing_activity_over_time(data):
    financing = data["results"]["financing"]
    time_steps = np.arange(financing.shape[1])

    # Count how many firms chose nonzero financing at each time
    financed = (financing > 0).sum(axis=0)

    plt.figure(figsize=(10,6))
    plt.plot(time_steps, financed, marker='o')
    plt.title("Number of Firms Using Financing Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Number of firms choosing financing")
    plt.grid(True)
    plt.show()

def plot_average_cash_vs_revenue(data):
    cash = data["results"]["X"]
    revenue = data["results"]["R"]

    avg_cash = np.mean(cash, axis=0)
    avg_revenue = np.mean(revenue, axis=0)
    time_steps = np.arange(cash.shape[1])

    plt.figure(figsize=(10,6))
    plt.plot(time_steps, avg_cash, label="Average Cash Balance")
    plt.plot(time_steps, avg_revenue, label="Average Revenue")
    plt.title("Average Cash and Revenue Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Millions")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_basis_betas(data):
    betas = data["results"]["betas"]
    time_steps = np.arange(betas.shape[0])

    plt.figure(figsize=(10,6))
    for i in range(betas.shape[1]):
        plt.plot(time_steps, betas[:, i], label=f"Beta {i}")
    
    plt.title("Evolution of Basis Function Betas Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Beta value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_bankruptcy_vs_financing_histogram(data):
    financing = data["results"]["financing"]
    bankruptcy = data["results"]["bankruptcy"]

    # Focus on financing decision at t=0 (or any other you want)
    t = 0
    financing_t = financing[:, t]
    bankruptcies_t = bankruptcy[:, t]

    # Make bins of financing
    unique_financing = np.unique(financing_t)
    
    bankruptcy_rate_per_financing = []
    for amount in unique_financing:
        mask = (financing_t == amount)
        if np.sum(mask) > 0:
            rate = np.mean(bankruptcies_t[mask])
            bankruptcy_rate_per_financing.append(rate)
        else:
            bankruptcy_rate_per_financing.append(np.nan)

    plt.figure(figsize=(10,6))
    plt.bar(unique_financing, np.array(bankruptcy_rate_per_financing) * 100, width=2)
    plt.title(f"Bankruptcy Rate vs Financing Amount (at time {t})")
    plt.xlabel("Financing Amount (Millions)")
    plt.ylabel("Bankruptcy Rate (%)")
    plt.grid(True)
    plt.show()

def plot_cash_vs_bankruptcy_heatmap(data, time_step=None):
    cash = data["results"]["X"]
    bankruptcy = data["results"]["bankruptcy"]

    if time_step is None:
        time_step = cash.shape[1] // 2  # Midpoint by default

    cash_now = cash[:, time_step]
    bankruptcy_now = bankruptcy[:, time_step]

    # Discretize cash balances into bins
    bins = np.linspace(np.min(cash_now), np.max(cash_now), 50)
    digitized = np.digitize(cash_now, bins)

    bankruptcy_prob = np.zeros(len(bins))
    for i in range(1, len(bins)):
        mask = digitized == i
        if np.sum(mask) > 0:
            bankruptcy_prob[i-1] = np.mean(bankruptcy_now[mask])
        else:
            bankruptcy_prob[i-1] = np.nan

    plt.figure(figsize=(10,6))
    plt.bar(bins, bankruptcy_prob * 100, width=(bins[1] - bins[0]))
    plt.title(f"Bankruptcy Probability by Cash Level (time {time_step})")
    plt.xlabel("Cash balance (Millions)")
    plt.ylabel("Bankruptcy Probability (%)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    for company in p.COMPANY_LIST:
        data = get_latest_simulation_results(company, version=4)
        print_main_results(data)
        print_all_parameters(data)
        plot_financing(data)
        plot_bankruptcy_timeline(data)

    # data = get_latest_simulation_results(329260, version=3)  
    # # print_main_results(data)
    # # print_all_parameters(data)
    # plot_revenue_and_cash(data)
    # # print_revenue_distributions(data)
    # # plot_firm_value_distribution(data)
    # plot_revenue_trajectories(data)
    # plot_bankruptcy_timeline(data)
    # # print_bankruptcy_matrix(data, max_rows=20)
    # plot_financing(data)
    # # plot_betas(data)
    # # plot_bankruptcy_rate_over_time(data)
    # # plot_financing_activity_over_time(data)
    # # plot_average_cash_vs_revenue(data)
    # # plot_basis_betas(data)
    # # plot_bankruptcy_vs_financing_histogram(data)
    # # plot_cash_vs_bankruptcy_heatmap(data, time_step=0)
    # # plot_bankruptcy_vs_financing_heatmap(data, time_step=0)
    # # plot_bankruptcy_vs_financing_heatmap(data, time_step=1)
    # # plot_bankruptcy_vs_financing_heatmap(data, time_step=2)
    # plot_bankruptcy_vs_financing_heatmap(data, time_step=3)
