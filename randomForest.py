import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#data
start = time.time()
df = pd.read_csv(r"data.csv")

# Guardamos copia para el heatmap ANTES
df_before = df.copy()

variables = [
    "FAMI_ESTRATOVIVIENDA", 
    "ESTU_GENERO",
    "ESTU_TIENEETNIA",
    "COLE_COD_DANE_ESTABLECIMIENTO",
    "COLE_MCPIO_UBICACION",
    "COLE_DEPTO_UBICACION",
    "ESTU_MCPIO_RESIDE",
    "ESTU_DEPTO_RESIDE",
    "COLE_NATURALEZA",
    "COLE_CALENDARIO",
    "COLE_CARACTER",
    "COLE_GENERO",
    "COLE_AREA_UBICACION",
    "COLE_JORNADA",
    "ESTU_DEDICACIONLECTURADIARIA",
    "ESTU_DEDICACIONINTERNET",
    "ESTU_HORASSEMANATRABAJA",
    "FAMI_EDUCACIONPADRE",
    "FAMI_EDUCACIONMADRE",
    "FAMI_PERSONASHOGAR",
    "PUNT_INGLES",
    "PERCENTIL_INGLES",
    "PUNT_C_NATURALES",
]

target = "PUNT_GLOBAL"

variables = [c for c in variables if c in df.columns]
df = df[variables + [target]].copy()

#Heatmap, before data cleansing
plt.figure(figsize=(10,6))
sns.heatmap(df_before[variables + [target]].isnull(), cbar=False)
plt.title("Heatmap de valores faltantes - ANTES de limpieza")
plt.show()


df = df.drop_duplicates()

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

#Heatmap, after data cleansing
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Heatmap de valores faltantes - DESPUÉS de limpieza")
plt.show()

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

X = df[variables]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42,
    stratify=pd.qcut(y, q=10, duplicates='drop')
)

#hyperparameters
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("====== RESULTADOS RANDOM FOREST ======")
print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R²:   {r2:.3f}")

#Graphs
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Predicho vs Real - Random Forest")
plt.grid(True)
plt.show()

end = time.time()
print(f"Tiempo total de ejecución: {end - start:.4f} segundos")