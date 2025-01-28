
import pandas as pd
#Query script for MySQL client
import pymysql
con = pymysql.connect(host='titlon.uit.no', 
                    user="x@ntnu.no", 
                    password="xxx", 
                    database='OSE')  
crsr=con.cursor()
crsr.execute("SET SESSION MAX_EXECUTION_TIME=60000;")
crsr.execute("""
	SELECT  * FROM `euronext`.`idx_sum_osl` 
	WHERE (`ISIN code` = 'NO0012513474') 
	AND year(`Effect date`) >= 2015
	ORDER BY `MIC`,`Instrument name`, `Effect date`
""")
r=crsr.fetchall()
df=pd.DataFrame(list(r), columns=[i[0] for i in crsr.description])
print(df)

df.to_csv('./data/titlon_accounting.csv', index=False)
