import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Generación de dataset con ruido sintético
data = {
    'edad': [25, 30, np.nan, 45, 22, 120, 35, 28, np.nan, 40], # 120 es atípico/erróneo
    'ingreso_mensual': [15000, 22000, 18000, np.nan, 12000, 950000, 25000, 21000, 19000, np.nan], # 950k outlier
    'scoring_credito': [650, 700, 680, 710, 590, 800, 670, 640, 620, 730]
}
df_dirty = pd.DataFrame(data)

# --- VISUALIZACIÓN INICIAL ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_dirty['ingreso_mensual'])
plt.title("Ingreso Mensual (Con Ruido)")

# 1. Tratamiento de Valores Atípicos por IQR sobre 'ingreso_mensual'
Q1 = df_dirty['ingreso_mensual'].quantile(0.25)
Q3 = df_dirty['ingreso_mensual'].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR

print(f"Límite Superior para Outliers en Ingreso: ${limite_superior:,.2f}")

# Reemplazar outliers extremos con NaN para tratarlos en la imputación
df_dirty.loc[df_dirty['ingreso_mensual'] > limite_superior, 'ingreso_mensual'] = np.nan
df_dirty.loc[df_dirty['edad'] > 100, 'edad'] = np.nan # Corrección de valor erróneo (Edad 120)

# 2. Imputación Múltiple Avanzada con KNNImputer (k=2)
imputer = KNNImputer(n_neighbors=2)
df_clean_array = imputer.fit_transform(df_dirty)
df_clean = pd.DataFrame(df_clean_array, columns=df_dirty.columns)

print("\n=== DATASET LIMPIO E IMPUTADO (KNN k=2) ===")
print(df_clean)

# --- COMPARATIVA CON MEDIA ---
df_mean_imputed = df_dirty.fillna(df_dirty.mean())
print("\n=== DATASET IMPUTADO POR MEDIA SIMPLE ===")
print(df_mean_imputed)

# --- VISUALIZACIÓN FINAL ---
plt.subplot(1, 2, 2)
sns.boxplot(y=df_clean['ingreso_mensual'])
plt.title("Ingreso Mensual (Limpio - KNN)")
plt.tight_layout()
plt.savefig('comparativa_outliers.png')
print("\nSe ha guardado el gráfico comparativo como 'comparativa_outliers.png'.")