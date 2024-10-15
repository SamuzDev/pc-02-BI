import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('breast_wisconsin_1.csv', sep=';')

X = data[['symmetry3']]  # Variable predictora
y = data['fractal_dimension3']  # Variable objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar resultados en consola
print(f'Error Cuadrático Medio (MSE): {mse}')
print(f'R²: {r2}')