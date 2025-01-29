from dotenv import load_dotenv
import os
import pandas as pd
import pymysql

# Load enviroment variables from .env file
# .env contains username and password: USERNAME=xxxxxx@ntnu.no and PASSWORD=xxxxxxxxxxxxxxxxx
load_dotenv()

#Query script for MySQL client
con = pymysql.connect(host='titlon.uit.no', 
                    user=os.getenv('USERNAME'), 
                    password=os.getenv('PASSWORD'), 
                    database='OSE')  
crsr=con.cursor()
crsr.execute("SET SESSION MAX_EXECUTION_TIME=60000;")
crsr.execute("""
	SELECT  * FROM `OSE`.`account` 
	WHERE `companyID` = 7795
	ORDER BY `Name`,`Year`
""")
r=crsr.fetchall()
df=pd.DataFrame(list(r), columns=[i[0] for i in crsr.description])
print(df)

df.to_csv('./data/titlon_accounting.csv', index=False)




#YOU NEED TO BE CONNECTED TO YOUR INSTITUTION VIA VPN, OR BE AT THE INSTITUTION, FOR THIS CODE TO WORK