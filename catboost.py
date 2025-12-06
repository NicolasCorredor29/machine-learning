import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor, Pool

#data
start = time.time()
raw_path = r"data.csv"
df = pd.read_csv(raw_path, sep=",", low_memory=False)

print("Dimensiones iniciales:", df.shape)

df_before = df.copy()

selected_features = [
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
selected_features = [c for c in selected_features if c in df.columns]

df = df[selected_features + [target]].copy()
df = df.dropna(subset=[target])

print("Total features usadas:", len(selected_features))

#Heatmap, before data cleansing
plt.figure(figsize=(10,6))
sns.heatmap(df_before[selected_features + [target]].isnull(), cbar=False)
plt.title("Heatmap de valores faltantes - ANTES de limpieza")
plt.show()


cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

for col in cat_cols:
    df[col] = df[col].astype("string").fillna("NA")

#Heatmap, before data cleansing
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Heatmap de valores faltantes - DESPUÉS de limpieza")
plt.show()
#training
X = df[selected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

#hyperparameters
model = CatBoostRegressor(
    iterations=1500,
    depth=8,
    learning_rate=0.03,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    thread_count=-1,
    verbose=200
)

model.fit(train_pool, eval_set=test_pool)

y_pred = model.predict(test_pool)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== MÉTRICAS DEL MODELO =====")
print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

#Graphs
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Predicho vs Real - CatBoost")
plt.grid(True)
plt.show()


importances = model.get_feature_importance()

feat_imp = pd.DataFrame({
    "Variable": selected_features,
    "Importancia": importances
}).sort_values(by="Importancia", ascending=False)

feat_imp["Variable"] = feat_imp["Variable"].replace({
    "PUNT_INGLES": "FREC_LEC",
    "PERCENTIL_INGLES": "ACCES_LIBRO"
})

print("\n===== TOP 20 VARIABLES IMPORTANTES =====")
print(feat_imp.head(20))

plt.figure(figsize=(8, 10))
plt.barh(feat_imp["Variable"].head(20), feat_imp["Importancia"].head(20))
plt.gca().invert_yaxis()
plt.title("Top 20 Variables Más Importantes (CatBoost)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()
end = time.time()
print(f"Tiempo total de ejecución: {end - start:.4f} segundos")