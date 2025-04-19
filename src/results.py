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
    filepath = f"simulation_outputs_latest/v1_{gvkey}_latest_sim_results.pkl"

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
    Plot the distribution of firm values from the simulation.
    '''
    # Hent data fra simuleringen
    X = data["results"]["X"]
    r_f = data["parameters"]["r_f"]
    T = data["parameters"]["T"]
    terminal_value = data["results"]["terminal_value"]

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


def plot_revenue_trajectories(data, num_trajectories=10):
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


if __name__ == "__main__":

    # for company in p.COMPANY_LIST:
    #     data = get_latest_simulation_results(company)
    #     print_main_results(data)

    # Ørsted
    data = get_latest_simulation_results(232646)
    print_main_results(data)
    print_all_parameters(data)
    plot_revenue_and_cash(data)
    print_revenue_distributions(data)
    plot_firm_value_distribution(data)
    plot_revenue_trajectories(data, num_trajectories=10)