import pandas as pd

data = pd.read_csv('aids_clinical_1-1.csv', sep=';')

data['homo_encoded'] = data['homo'].astype('category').cat.codes

correlation = data['preanti'].corr(data['homo_encoded'])

if correlation > 0:
    print(f"Correlación positiva: {correlation}")
elif correlation < 0:
    print(f"Correlación negativa: {correlation}")
else:
    print(f"No hay correlación: {correlation}")