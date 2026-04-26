import os
from dotenv import load_dotenv
import boto3
import pandas as pd
import io

load_dotenv()

s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('YC_ENDPOINT'),
    region_name=os.getenv('YC_REGION'),
    aws_access_key_id=os.getenv('YC_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('YC_SECRET_ACCESS_KEY'),
)

# Проверим данные для 2020-02-02
key = 'raw/klines/symbol=ADAUSDT/interval=1m/date=2020-02-02/data.parquet'

try:
    response = s3.get_object(Bucket='binance-data-downloader', Key=key)
    df = pd.read_parquet(io.BytesIO(response['Body'].read()))

    print('Successfully loaded data for 2020-02-02')
    print(f'Shape: {df.shape}')
    print(f'First timestamp: {df["timestamp"].min()}')
    print(f'Last timestamp: {df["timestamp"].max()}')
    print(f'Expected rows: 1440, Actual rows: {len(df)}')

    if len(df) == 1440:
        print('CORRECT: Data has been restored successfully!')
    else:
        print('ERROR: Data restoration failed!')

    # Проверим первые несколько минут
    print()
    print('First 5 minutes:')
    print(df.head()[['timestamp', 'open', 'high', 'low', 'close', 'volume']])

except Exception as e:
    print(f'Error loading data: {e}')