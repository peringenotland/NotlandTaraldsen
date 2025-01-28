import nasdaqdatalink

# Set your API key
nasdaqdatalink.ApiConfig.api_key = '1yW9NMZanZKaCDczxVD3'

data = nasdaqdatalink.get_table('MER/F1', compnumber=["39102"], paginate=True)

print(data.columns)