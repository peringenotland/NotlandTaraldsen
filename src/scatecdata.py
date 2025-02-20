import pandas as pd

file_path = "C:\\Users\\taral\\Documents\\Dev\\Master\\NotlandTaraldsen\\compustat_scatec.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

print(df[['revtq','cogsq', 'xoprq']])