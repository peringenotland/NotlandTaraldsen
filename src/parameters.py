import numpy as np
import pandas as pd


csv_path = 'data/all_companies_quarterly.csv'
companies = [103342, 225094, 225597, 232646, 245628, 318456, 328809, 329260] # TODO: ENEL mangler
# [103342 SSE, 225094 VESTAS, 225597 FORTUM, 232646 ORSTED, 245628 NORDEX, 318456 SCATEC, 328809 NEOEN, 329260 ENCAVIS, 349408 (FEIL), 295785 ENEL]

# Load data from CSV file
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    raise FileNotFoundError(f"Error reading CSV file at {csv_path}: {e}")



def get_R_0(gvkey):
    '''
    Initial Revenue
    TODO: Burde ha en plan for å deale med seasonality
    '''
    return df[df['gvkey'] == gvkey]['revtq'].iloc[-1] # Last quarter revenue



def get_L_0(gvkey):
    '''
    Initial Loss Carryforward
    denne finnes ikke i compustat global
    Enten må vi regne ut fra den dataen vi har, eller finne manuelt i company reports, eller bytte til North American compustat
    TODO: Se på dette.
    '''
    # return df[df['gvkey'] == gvkey]['txdb'].iloc[-1]
    return 0



def get_X_0(gvkey):
    '''
    Initial Cash Balance
    '''
    return df[df['gvkey'] == gvkey]['cheq'].iloc[-1]


def get_mu_0():
    """
    Initial expected rate of growth in revenues (mu_0)
    TODO: Burde ha en plan for å deale med seasonality
    TODO: Markedssnitt 
    """
    # Get the most recent growth rate
    growth_rates = []
    for company in companies:
        q12024 = df[(df['gvkey'] == company) & (df['fqtr'] == 1) & (df['fyearq'] == 2024)]['revtq'].values[0]
        q12020 = df[(df['gvkey'] == company) & (df['fqtr'] == 1) & (df['fyearq'] == 2020)]['revtq'].values[0]
        # print(f'q12024: {q12024}, q12020: {q12020}')
        cagr = ((q12024 / q12020) ** (1/16)) - 1
        # print(f'Company: {company}, CAGR: {cagr}')
        growth_rates.append(cagr)
    # print(f'Growth rates: {growth_rates}')
    return np.mean(growth_rates)


def get_sigma_0():
    """
    Initial volatility of revenues (sigma_0) - Mean of individual company standard deviations.
    TODO: Adjust for seasonality in revenue growth. Nå blir det rekna per quarter
    """
    company_volatilities = []

    for company in companies:
        # Extract quarterly revenue data for each company
        revenues = df[df['gvkey'] == company].sort_values(by=['fyearq', 'fqtr'])[['fyearq', 'fqtr', 'revtq']].copy()

        # Calculate quarterly revenue growth rates
        revenues.loc[:, 'growth_rate'] = revenues['revtq'].pct_change()

        # Drop NaN values (first row will have NaN)
        growth_rates = revenues['growth_rate'].dropna().values

        # Compute standard deviation only if there are enough data points
        if len(growth_rates) > 1:  
            company_volatilities.append(np.std(growth_rates, ddof=1))  # Sample std dev

    # Compute the average volatility across companies (ignore NaN values)
    return np.nanmean(company_volatilities) if company_volatilities else np.nan

import numpy as np


def get_eta_0(stock_prices, frequency='daily'):
    """
    Estimates eta_0, the initial volatility of expected growth rate in revenues.
    Uses historical stock price log-returns to infer volatility.
    """
    log_returns = np.log(stock_prices / np.roll(stock_prices, 1))[1:]  # Compute log-returns
    volatility = np.std(log_returns, ddof=1)  # Sample standard deviation

    # Annualize based on frequency
    if frequency == 'daily':
        volatility *= np.sqrt(252)
    elif frequency == 'monthly':
        volatility *= np.sqrt(12)

    return volatility


def get_rho():
    """
    Correlation between revenue and growth rate (rho)
    SchwartzMoon antar 0.0
    "Estimated from past company or cross-sectional data"

    Consider the situation in which the firm is in a competitive environment such that 
    increases in profit margins (i.e. decreases in variable costs) are associated with 
    decreases in growth rates in revenues. This implies a positive correlation between
    variable costs and growth rates in revenues. We run the program with the same
    data described above, but assuming a correlation of 0.5. The market price of risk
    and the volatility of the growth rate had to be adjusted slightly to match the volatility
    and the beta of the stock. The model price decreased only slightly to $21.89.
    The effect of this correlation on prices increases, however, with the volatility of
    variable costs.
    """
    return 0.0


def get_mu_mean():
    """
    Mean-reversion level for growth rate (mu_mean)
    """
    return 0.015

def get_sigma_mean():    
    """
    Mean-reversion level for volatility (sigma_mean)
    """
    return 0.05

def get_taxrate():
    """
    Corporate tax rate (taxrate)
    """
    return 0.35

def get_r_f():
    """
    Risk-free rate (r_f)
    """
    return 0.05

def get_kappa_mu():
    """
    Mean-reversion speed for expected growth rate (kappa_mu)
    """
    return 0.07

def get_kappa_sigma():
    """
    Mean-reversion speed for volatility (kappa_sigma)
    """
    return 0.07






def print_pivot_table(value='revtq'):
    print('\nPIVOT TABLE:')
    df_pivot = df.pivot(index=['fyearq', 'fqtr'], columns='gvkey', values=value)
    print(df_pivot)
    print('\n')
    return df_pivot


def main():
    gvkey = 103342

    print_pivot_table('revtq')

    mu_0 = get_mu_0()
    print(f'mu_0: {mu_0}')

    

main()