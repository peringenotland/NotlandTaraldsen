import pandas as pd

file_path = "C:\\Users\\taral\\Documents\\Dev\\Master\\NotlandTaraldsen\\titlon_accounting.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Filter for the specific company and years up to 2022
company_id = 12720  # Replace with your desired company ID
filtered_df = df[(df['companyID'] == company_id) & (df['Year'] == 2022) & (df['account_type'] == 'RESULTATREGNSKAP') ]
#revenue_df = filtered_df[filtered_df['description'].str.contains("inntekter", case=False, na=False)]

print(filtered_df[['Name', 'Year', 'description', 'Value']])