from faker import Faker
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuración de Faker para México y fijación de semillas para reproducibilidad
fake = Faker('es_MX')
Faker.seed(42)
np.random.seed(42)

# 1. Generación programática de dataset con Faker
registros = []
for i in range(1000):
    registros.append({
        'id_cliente': i + 1,
        'nombre': fake.name(),
        'email': fake.email(),
        # --- EXTENSIÓN: Agregando dos variables sintéticas adicionales ---
        'direccion': fake.address().replace('\n', ', '), 
        'telefono': fake.phone_number(),
        'estado_civil': np.random.choice(['Soltero', 'Casado', 'Divorciado'], p=[0.5, 0.4, 0.1]),
        'ingreso_anual': round(np.random.uniform(100000, 1000000), 2),
        'churn': np.random.choice([0, 1], p=[0.8, 0.2]) # 20% tasa de abandono
    })

df_sintetico = pd.DataFrame(registros)

# --- EXTENSIÓN: Codificación de variables categóricas ---
# Convertimos 'estado_civil' a columnas numéricas binarias (One-Hot Encoding)
df_sintetico = pd.get_dummies(df_sintetico, columns=['estado_civil'], dtype=int)

print("=== MUESTRA DEL DATASET SINTÉTICO (CON ONE-HOT ENCODING) ===")
# Mostramos algunas columnas para verificar la codificación y los nuevos datos
print(df_sintetico[['id_cliente', 'telefono', 'ingreso_anual', 'estado_civil_Casado', 'estado_civil_Divorciado', 'estado_civil_Soltero', 'churn']].head())

# 2. Selección de Datos Objetivo (Target) y Características (Features)
# Excluimos las variables identificadoras de texto plano y el target (churn)
X = df_sintetico.drop(columns=['id_cliente', 'nombre', 'email', 'direccion', 'telefono', 'churn'])
y = df_sintetico['churn']

# 3. Partición Estratificada (Train 80%, Test 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\n=== RESULTADOS DE LA PARTICIÓN ESTRATIFICADA ===")
print(f"Proporción Churn en Dataset Completo: {y.mean():.4f}")
print(f"Dimensiones Train Set: {X_train.shape} | Proporción Churn: {y_train.mean():.4f}")
print(f"Dimensiones Test Set:  {X_test.shape} | Proporción Churn: {y_test.mean():.4f}")