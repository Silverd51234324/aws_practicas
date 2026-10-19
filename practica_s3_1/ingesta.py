import boto3
import pandas as pd
import json
import io

# 1. Configuración de cliente Boto3 para conexión con AWS Academy S3
s3_client = boto3.client('s3', region_name='us-east-1')
BUCKET_NAME = 'aws-academy-datacenter-bucket-23-02-002'

def cargar_y_perfilar_s3(file_key, is_json=False):
    """Descarga archivos desde S3 y genera un perfilamiento básico de tipos de datos."""
    print(f"=== Procesando objeto S3: {file_key} ===")
    
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
    body = response['Body'].read()
    
    if is_json:
        data_json = json.loads(body.decode('utf-8'))
        df = pd.json_normalize(data_json)
    else:
        df = pd.read_csv(io.BytesIO(body))

    # Adecuación para conversión de fechas
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Adecuación para rellenar valores nulos (NaN) en sensores
    df = df.fillna(0)
        
    print(f"Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("\nPerfil de Tipos de Datos e Identificación de Tipos:")
    print(df.dtypes)
    print("\nResumen de Estadísticos Descriptivos (Cuantitativos):")
    print(df.describe().T)
    
    return df

# Ejemplo de ejecución dentro del entorno virtual
if __name__ == "__main__":
    # Requiere credenciales activas de AWS Academy
    df_transacciones = cargar_y_perfilar_s3('raw/transacciones.csv')
    df_logs = cargar_y_perfilar_s3('raw/logs_m2m.json', is_json=True)