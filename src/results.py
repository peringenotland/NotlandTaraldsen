import pickle
import parameters as p

def get_latest_simulation_results(gvkey):
    """
    Load simulation results from a pickle file.

    Parameters:
    - gvkey (str): The gvkey of the firm.

    Returns:
    - dict: A dictionary containing the simulation results.
    """
    # Path to the file you saved earlier
    filepath = f"simulation_outputs_latest/{gvkey}_latest_sim_results.pkl"

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



if __name__ == "__main__":
    for company in p.COMPANY_LIST:
        data = get_latest_simulation_results(company)
        print_main_results(data)
