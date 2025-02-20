from dotenv import load_dotenv
import os
import pandas as pd
import pymysql

# Load enviroment variables from .env file in src folder
# .env contains: 
# TITLON_USERNAME=xxxxxx@ntnu.no
# TITLON_PASSWORD=xxxxxxxxxxxxxxxxx

load_dotenv()

#Query script for MySQL client
con = pymysql.connect(host='titlon.uit.no', 
                    user=os.getenv('TITLON_USERNAME'), 
                    password=os.getenv('TITLON_PASSWORD'), 
                    database='OSE')  
crsr=con.cursor()
crsr.execute("SET SESSION MAX_EXECUTION_TIME=600000;")
crsr.execute("""
	SELECT  * FROM `OSE`.`account`
	ORDER BY `Name`,`Year`,`account_number`
""")
r=crsr.fetchall()
df=pd.DataFrame(list(r), columns=[i[0] for i in crsr.description])
print(df)

df.to_csv('latest_output.csv', index=False)


#YOU NEED TO BE CONNECTED TO YOUR INSTITUTION VIA VPN, OR BE AT THE INSTITUTION, FOR THIS CODE TO WORK

file_path = "C:\Users\taral\Documents\Dev\Master\NotlandTaraldsen\titlon_accounting.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Filter for the specific company and years up to 2022
company_id = 7795  # Replace with your desired company ID
filtered_df = df[(df['companyID'] == company_id) & (df['Year'] <= 2022)]
revenue_df = filtered_df[filtered_df['description'].str.contains("revenue", case=False, na=False)]
print(revenue_df[['Name', 'Year', 'description', 'Value', 'ID']])