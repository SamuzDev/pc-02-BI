import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

file_state = 'state_x78-1.csv'
data_state = pd.read_csv(file_state)

data_state_cleaned = data_state.iloc[:, 0].str.split(';', expand=True)
data_state_cleaned.columns = ['habitantes', 'ingresos', 'analfabetismo', 'esp_vida', 
                              'asesinatos', 'universitarios', 'heladas', 'area', 'densidad_pobl']
                              
data_state_cleaned['ingresos'] = pd.to_numeric(data_state_cleaned['ingresos'], errors='coerce')
data_state_cleaned['esp_vida'] = pd.to_numeric(data_state_cleaned['esp_vida'], errors='coerce')
data_state_cleaned = data_state_cleaned.dropna(subset=['ingresos', 'esp_vida'])

X_train, X_test, y_train, y_test = train_test_split(data_state_cleaned[['ingresos']], 
                                                    data_state_cleaned['esp_vida'], 
                                                    test_size=0.18, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")
