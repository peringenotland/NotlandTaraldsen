import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FUNDAMENTALS_CSV_PATH = 'data/all_companies_quarterly.csv'
FUNDAMENTALS_Y2D_CSV_PATH = 'data/all_companies_y2d.csv'
STOCK_PRICES_CSV_PATH = 'data/stock_prices.csv'
STOXX_600_CSV_PATH = 'data/stoxx600_monthly.csv'
COMPANY_LIST = [103342, 225094, 225597, 232646, 245628, 318456, 328809, 329260] # TODO: ENEL mangler
COMPANY_DICT = {'SSE': 103342, 'VESTAS': 225094, 'FORTUM': 225597, }
# [103342 SSE, 225094 VESTAS, 225597 FORTUM, 232646 ORSTED, 245628 NORDEX, 318456 SCATEC, 328809 NEOEN, 329260 ENCAVIS, 349408 (FEIL), 295785 ENEL]

# Load fundamentals data from CSV file
try:
    FUNDAMENTALS = pd.read_csv(FUNDAMENTALS_CSV_PATH)
except Exception as e:
    raise FileNotFoundError(f"Error reading CSV file at {FUNDAMENTALS_CSV_PATH}: {e}")

try:
    FUNDAMENTALS_Y2D = pd.read_csv(FUNDAMENTALS_Y2D_CSV_PATH)
except Exception as e:
    raise FileNotFoundError(f"Error reading CSV file at {FUNDAMENTALS_Y2D_CSV_PATH}: {e}")


# Load stock prices
try:
    STOCK_PRICES = pd.read_csv(STOCK_PRICES_CSV_PATH, low_memory=False)
    STOCK_PRICES = STOCK_PRICES[STOCK_PRICES['iid'] == '01W']   # Filter for '01W' iid
except Exception as e:
    raise FileNotFoundError(f"Error reading stock prices CSV file at {STOCK_PRICES_CSV_PATH}: {e}")

try:
    STOXX_600 = pd.read_csv(STOXX_600_CSV_PATH)
    STOXX_600['datadate'] = pd.to_datetime(STOXX_600['datadate'], errors='coerce')
    STOXX_600 = STOXX_600.sort_values('datadate')
    prices = STOXX_600.set_index('datadate')['prccm']

    # Resample to quarterly prices (using last price of each quarter)
    quarterly_prices = prices.resample("QE").last()

    # Calculate log returns
    STOXX_QUARTERLY = np.log(quarterly_prices / quarterly_prices.shift(1)).dropna()

    # Convert to DataFrame and add 'fyearq' and 'fqtr'
    STOXX_QUARTERLY = STOXX_QUARTERLY.to_frame('log_return_market')
    STOXX_QUARTERLY['fyearq'] = STOXX_QUARTERLY.index.year
    STOXX_QUARTERLY['fqtr'] = STOXX_QUARTERLY.index.quarter

    # Calculate standard deviation of returns
    SIGMA_MARKET = STOXX_QUARTERLY['log_return_market'].std()
except Exception as e:
    raise FileNotFoundError(f"Error reading CSV file at {STOXX_600_CSV_PATH}: {e}")

