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
crsr.execute("SET SESSION MAX_EXECUTION_TIME=60000;")
crsr.execute("""
	SELECT  * FROM `OSE`.`account`
	ORDER BY `Name`,`Year`
""")
r=crsr.fetchall()
df=pd.DataFrame(list(r), columns=[i[0] for i in crsr.description])
print(df)

df.to_csv('titlon_accounting.csv', index=False)


#YOU NEED TO BE CONNECTED TO YOUR INSTITUTION VIA VPN, OR BE AT THE INSTITUTION, FOR THIS CODE TO WORK