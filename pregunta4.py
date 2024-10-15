import pandas as pd

file_ingreso = 'ingreso-1.csv'
data_ingreso = pd.read_csv(file_ingreso)

correlation_ingreso_horas = data_ingreso['ingreso'].corr(data_ingreso['horas'])

print(f"Correlación entre ingreso y horas: {correlation_ingreso_horas}")