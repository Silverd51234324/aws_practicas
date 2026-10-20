import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import boto3
import io

def pipeline_etl_completo():
    print("=== 1. FASE DE EXTRACCIÓN (Extract) ===")
    # Conexión a AWS S3
    s3_client = boto3.client('s3', region_name='us-east-1')
    BUCKET_NAME = 'aws-academy-datacenter-bucket-23-02-002' # <--- ¡SUSTITUYE POR TU BUCKET DE LA PRÁCTICA 1!
    
    # Extensión: Extraer datos reales desde S3 (zona raw)
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key='raw/transacciones.csv')
        df_raw = pd.read_csv(response['Body'])
        print("-> Datos extraídos correctamente de S3 (raw/transacciones.csv)")
    except Exception as e:
        print(f"Error extrayendo de S3. Verifica tu BUCKET_NAME y credenciales: {e}")
        return
    print("\n=== 2. FASE DE TRANSFORMACIÓN (Transform) ===")
    # Discretización (Binning) sobre el 'monto' de la transacción
    df_raw['categoria_monto'] = pd.cut(
        df_raw['monto'],
        bins=[0, 200, 1000, 10000],
        labels=['Bajo', 'Medio', 'Alto']
    )
    
    # Normalización Min-Max del monto
    scaler = MinMaxScaler()
    df_raw['monto_normalizado'] = scaler.fit_transform(df_raw[['monto']])
    
    # Feature Engineering (Ingeniería de Características)
    df_raw['score_promocional'] = df_raw['monto_normalizado'] * 100
    
    print("-> Dataset Transformado (Muestra):")
    print(df_raw[['id_transaccion', 'monto', 'categoria_monto', 'monto_normalizado', 'score_promocional']].head())

    print("\n=== 3. FASE DE CARGA (Load) ===")
    # Carga de resultado transformado hacia formato Parquet en memoria
    buffer = io.BytesIO()
    df_raw.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    # Extensión: Subida a AWS S3 en la zona 'processed'
    KEY_S3 = 'processed/etl_output.parquet'
    try:
        s3_client.put_object(Bucket=BUCKET_NAME, Key=KEY_S3, Body=buffer.getvalue())
        print(f"-> Éxito: Archivo Parquet comprimido cargado correctamente en la zona 'processed' de S3 ({KEY_S3}).")
    except Exception as e:
        print(f"Error al cargar a S3: {e}")

if __name__ == "__main__":
    pipeline_etl_completo()