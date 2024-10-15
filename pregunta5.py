import pandas as pd
import numpy as np

data = {
    'x1': [1, 2, 8, 4, 1, 6],
    'x2': [100, 2, 3, 1, 2, 7],
    'x3': [34, None, None, None, 27, 44],
    'x4': [102, 121, 343, None, 121, 125],
    'x5': [125, None, 215, None, 121, 125],
    'x6': [15, None, 14, None, 12, None]
}

df = pd.DataFrame(data)

print("DataFrame original:")
print(df)

df_imputed = df.apply(lambda col: col.fillna(col.median()))

print("\nDataFrame después de la imputación con la mediana:")
print(df_imputed)