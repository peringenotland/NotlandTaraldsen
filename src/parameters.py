import numpy as np
import pandas as pd


FUNDAMENTALS_CSV_PATH = 'data/all_companies_quarterly.csv'
STOCK_PRICES_CSV_PATH = 'data/stock_prices.csv'
COMPANY_LIST = [103342, 225094, 225597, 232646, 245628, 318456, 328809, 329260] # TODO: ENEL mangler
# [103342 SSE, 225094 VESTAS, 225597 FORTUM, 232646 ORSTED, 245628 NORDEX, 318456 SCATEC, 328809 NEOEN, 329260 ENCAVIS, 349408 (FEIL), 295785 ENEL]

# Load fundamentals data from CSV file
try:
    FUNDAMENTALS = pd.read_csv(FUNDAMENTALS_CSV_PATH)
except Exception as e:
    raise FileNotFoundError(f"Error reading CSV file at {FUNDAMENTALS_CSV_PATH}: {e}")

# Load stock prices
try:
    STOCK_PRICES = pd.read_csv(STOCK_PRICES_CSV_PATH)
    STOCK_PRICES = STOCK_PRICES[STOCK_PRICES['iid'] == '01W']   # Filter for '01W' iid
except Exception as e:
    raise FileNotFoundError(f"Error reading stock prices CSV file at {STOCK_PRICES_CSV_PATH}: {e}")



def get_R_0(gvkey, df=FUNDAMENTALS):
    '''
    Initial Revenue
    TODO: Burde ha en plan for å deale med seasonality
    '''
    return df[df['gvkey'] == gvkey]['revtq'].iloc[-1] # Last quarter revenue



def get_L_0(gvkey, df=FUNDAMENTALS):
    '''
    Initial Loss Carryforward
    denne finnes ikke i compustat global
    Enten må vi regne ut fra den dataen vi har, eller finne manuelt i company reports, eller bytte til North American compustat
    TODO: Se på dette.
    '''
    # return df[df['gvkey'] == gvkey]['txdb'].iloc[-1]
    return 0



def get_X_0(gvkey, df=FUNDAMENTALS):
    '''
    Initial Cash Balance
    '''
    return df[df['gvkey'] == gvkey]['cheq'].iloc[-1]


def get_mu_0(df=FUNDAMENTALS):
    """
    Initial expected rate of growth in revenues (mu_0)
    TODO: Burde ha en plan for å deale med seasonality
    TODO: Markedssnitt -> Denne tar snittet av vekstratene til selskapene i COMPANY_LIST
    TODO: Tenke litt på om det er lurt å bruke snittet av vekstratene til selskapene i COMPANY_LIST, eller om det er bedre å bruke vekstraten til hvert spesifikt selskap.
    TODO: Damodaran har 15.73% i average for renewable energy de siste 5 årene.
    """
    # Get the most recent growth rate
    growth_rates = []
    for company in COMPANY_LIST:
        q12024 = df[(df['gvkey'] == company) & (df['fqtr'] == 1) & (df['fyearq'] == 2024)]['revtq'].values[0]
        q12020 = df[(df['gvkey'] == company) & (df['fqtr'] == 1) & (df['fyearq'] == 2020)]['revtq'].values[0]
        # print(f'q12024: {q12024}, q12020: {q12020}')
        cagr = ((q12024 / q12020) ** (1/16)) - 1
        # print(f'Company: {company}, CAGR: {cagr}')
        growth_rates.append(cagr)
    # print(f'Growth rates: {growth_rates}')
    return np.mean(growth_rates)


def get_sigma_0(df=FUNDAMENTALS):
    """
    Initial volatility of revenues (sigma_0) - Mean of all company quarterly standard deviations.
    TODO: Adjust for seasonality in revenue growth. Nå blir det rekna per quarter
    """
    company_volatilities = []

    for company in COMPANY_LIST:
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



def get_eta_0(gvkey, frequency='quarterly', df=STOCK_PRICES):
    """
    Estimates eta_0, the initial volatility of expected growth rate in revenues.
    Uses historical stock price log-returns to infer volatility.
    Vi bruker iid=01W, som er hvilken issue av aksjen som er i bruk.
    TODO: Sjekke Dixit-Pindyck om at dette er normal måte å gjøre dette på.
    """
    firm_data = df[df['gvkey'] == gvkey].dropna(subset=['prccd'])
    firm_data['datadate'] = pd.to_datetime(df['datadate'], errors='coerce')
    firm_data = firm_data.sort_values('datadate')
    print(firm_data)

    if firm_data.empty:
        raise ValueError(f"No price data found for gvkey {gvkey}")

    prices = firm_data['prccd'].values

    if len(prices) < 2:
        raise ValueError(f"Not enough price data to compute returns for gvkey {gvkey}")

    # Compute log returns
    log_returns = np.diff(np.log(prices))

    # Calculate sample standard deviation of daily returns
    volatility = np.std(log_returns, ddof=1)

    # Scale to desired frequency
    if frequency == 'daily':
        return volatility
    elif frequency == 'monthly':
        return volatility * np.sqrt(21)   # ~21 trading days per month
    elif frequency == 'quarterly':
        return volatility * np.sqrt(63)   # ~63 trading days per quarter
    elif frequency == 'annual':
        return volatility * np.sqrt(252)
    else:
        raise ValueError("Frequency must be 'daily', 'monthly', 'quarterly', or 'annual'.")


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
    return 0.0 # Kan la denne være 0


def get_mu_mean():
    """
    Mean-reversion level for growth rate (mu_mean)
    
    Schwartz Moon (2000):
    Rate of growth in revenues for a stable company in the same industry as the company being valued.
    
    Schwartz Moon (2001):
    Figure 2 shows distributions of  the rate  of  growth in revenues implied from 
    these parameters in one, three and ten  years. Note that the distribution shrinks and 
    moves left as time increases; it converges to a constant 0.05 at infinity.
    """
    return 0.015 # dette blir 6% i året, noe som er rimelig. Ref NotlandTaraldsen

def get_sigma_mean():    
    """
    Mean-reversion level for volatility (sigma_mean)
    TODO: Ha med i discussion: Is this assumption reasonable?
    For a stable renewable energy company, yes—it can be quite reasonable:

    Renewable energy producers (especially large-scale utilities) often face relatively low revenue volatility once operational, due to:
    Long-term power purchase agreements (PPAs)
    Stable demand
    Government subsidies or regulated pricing
    """
    return 0.05

def get_taxrate(gvkey, df=FUNDAMENTALS):
    """
    Corporate tax rate (taxrate)
    """
    firm_data = df[df['gvkey'] == gvkey]

    if firm_data.empty:
        return None  # or np.nan
    
    try:
        taxes = firm_data['txtq'].iloc[-1]
        pretax_income = firm_data['piq'].iloc[-1]
        if pretax_income <= 0:
            return 0.24  # Default tax rate if pretax income is zero or negative
        taxrate = taxes / pretax_income
        if taxrate <= 0.15:  # If the calculated tax rate is too low, use a default value
            taxrate = 0.24  
        return taxrate
    except (IndexError, KeyError, ZeroDivisionError):
        return None



def get_r_f():
    """
    Risk-free rate (r_f)
    TODO: vi har valgt 25Y, altså til 2050, pga netzero target og valuation horizon.
    """
    return 0.03055489 # European 25Y AAA bond yield as of 2024-01-01 (3.055489%)


def get_kappa_mu(convergence=0.95):
    """
    Mean-reversion speed for expected growth rate (kappa_mu)
    """
    kappa = -1 * np.log(1 - convergence) / 100
    return kappa

def get_kappa_sigma(convergence=0.95):
    """
    Mean-reversion speed for volatility (kappa_sigma)
    """
    kappa = -1 * np.log(1 - convergence) / 100
    return kappa

def get_kappa_eta(convergence=0.95):
    """
    Mean-reversion speed for expected growth rate volatility (kappa_eta)
    """
    kappa = -1 * np.log(1 - convergence) / 100
    return kappa


def get_gamma_0(gvkey, df=FUNDAMENTALS, n_quarters=8):
    """
    Calculate the ratio of total costs to revenues for the last quarters of the firm.
    Without Depreciation and Interest. (EBITDA margin) Need to include Dep in montecarlo simulation.
    """
    firm_data = df[df['gvkey'] == gvkey].sort_values(by='datadate')

    if firm_data.empty:
        return None

    # Get the last `n_quarters` of data
    recent_data = firm_data.tail(n_quarters)

    ratios = []
    for _, row in recent_data.iterrows():
        revenue = row.get('saleq', None)
        cogs = row.get('cogsq', 0)
        cogs = 0 if pd.isna(cogs) else cogs
        sga = row.get('xsgaq', 0)
        sga = 0 if pd.isna(sga) else sga
        # depr = row.get('dpq', 0) or 0
        # interest = row.get('xintq', 0) or 0

        if pd.isna(revenue) or revenue == 0:
            continue  # skip invalid or missing quarters

        operating_costs = cogs + sga # + depr + interest
        ratio = operating_costs / revenue
        ratios.append(ratio)

    if not ratios:
        return None

    return sum(ratios) / len(ratios)


def get_gamma_mean():
    """
    Mean-reversion level for the ratio of total cost to revenues (gamma_mean)
    """
    gammas = []
    for company in COMPANY_LIST:
        gamma = get_gamma_0(company)
        if gamma is not None:
            gammas.append(gamma)
    return np.nanmean(gammas) if gammas else 0.75

def get_kappa_gamma(convergence=0.95):
    """
    Mean-reversion speed for the ratio of total cost to revenues (kappa_gamma)
    """
    kappa = -1 * np.log(1 - convergence) / 100
    return kappa

def get_phi_0():
    """
    Initial volatility of the ratio of total cost to revenues (phi_0)
    """
    return 0.0

def get_phi_mean():
    """
    Mean-reversion level for the volatility of the ratio of total cost to revenues (phi_mean)
    """
    return 0.0

def get_kappa_phi(convergence=0.95):
    """
    Mean-reversion speed for the volatility of the ratio of total cost to revenues (kappa_phi)
    """
    kappa = -1 * np.log(1 - convergence) / 100
    return kappa



def get_lambda_1():
    """
    Market price of risk for the revenue factor (lambda_1)
    """
    return 0.01

def get_lambda_2():
    """
    Market price of risk for the expected rate of growth in revenues factor (lambda_2)
    """
    return 0.0

def get_T():
    """
    Time horizon in years (T)
    """   
    return 25

def get_dt():
    """
    Time step (dt)
    """
    return 1

def get_M():
    """
    Exit multiple (M)
    """
    return 10

def get_simulations():
    """
    Number of Monte Carlo runs (simulations)
    """
    return 100000

def get_num_steps():
    """
    Number of steps in the simulation
    """
    return get_T() * 4 + 1



def print_pivot_table(value='revtq', df=FUNDAMENTALS):
    print('\nPIVOT TABLE:')
    df_pivot = df.pivot(index=['fyearq', 'fqtr'], columns='gvkey', values=value)
    print(df_pivot)
    print('\n')
    return df_pivot


def main():
    gvkey = 103342
    # [103342 SSE, 225094 VESTAS, 225597 FORTUM, 232646 ORSTED, 245628 NORDEX, 318456 SCATEC, 328809 NEOEN, 329260 ENCAVIS, 349408 (FEIL), 295785 ENEL]



    eta_0 = get_eta_0(gvkey)
    print(f'eta_0: {eta_0}')

    print(f'Initial growth rate (mu_0): {get_mu_0()}')
    print(f'Initial volatility of revenues (sigma_0): {get_sigma_0()}')

    print(f'Tax rate (taxrate): {get_taxrate(gvkey)}')
    print(f'Kappa mu (kappa_mu): {get_kappa_mu()}')
    print(f'Initial gamma (gamma_0): {get_gamma_0(gvkey)}')
    print(f'Gamma mean (gamma_mean): {get_gamma_mean()}')
    

main()