import pickle
import parameters as p
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_latest_simulation_results(gvkey):
    """
    Load simulation results from a pickle file.

    Parameters:
    - gvkey (str): The gvkey of the firm.

    Returns:
    - dict: A dictionary containing the simulation results.
    """
    # Path to the file you saved earlier
    filepath = f"simulation_outputs_latest/v2_{gvkey}_latest_sim_results.pkl"

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
    # for i in range(200):
    #     plt.plot(X[i, :], alpha=0.05, color='green')

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
    Plot when bankruptcies occur over time.

    Shows the number of bankruptcies in each quarter.
    """
    bankruptcy = data["results"]["bankruptcy"]  # shape (simulations, num_steps)

    num_steps = bankruptcy.shape[1]
    time = np.arange(num_steps)
    bankruptcies_per_step = np.sum(bankruptcy, axis=0)

    plt.figure(figsize=(10, 5))
    plt.bar(time, bankruptcies_per_step, color='salmon', alpha=0.8)
    plt.xlabel("Quarter")
    plt.ylabel("Number of Bankruptcies")
    plt.title("Bankruptcies Over Time")
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



if __name__ == "__main__":

    # for company in p.COMPANY_LIST:
    #     data = get_latest_simulation_results(company)
    #     print_main_results(data)

    
    data = get_latest_simulation_results(103342)
    print_main_results(data)
    print_all_parameters(data)
    plot_revenue_and_cash(data)
    print_revenue_distributions(data)
    plot_firm_value_distribution(data)
    plot_revenue_trajectories(data)
    plot_bankruptcy_timeline(data)
    print_bankruptcy_matrix(data, max_rows=20)

    # TODO: Finn ut bankruptcy.