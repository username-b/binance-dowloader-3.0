import os
from datetime import datetime, timedelta
from collections import Counter
import boto3
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('YC_ENDPOINT'),
    region_name=os.getenv('YC_REGION'),
    aws_access_key_id=os.getenv('YC_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('YC_SECRET_ACCESS_KEY'),
)

bucket = 'binance-data-downloader'
prefix = 'raw/aggTrades/symbol=ADAUSDT/'

print('Listing objects under', prefix)
objects = []
kwargs = {'Bucket': bucket, 'Prefix': prefix}
while True:
    resp = s3.list_objects_v2(**kwargs)
    contents = resp.get('Contents', [])
    objects.extend(contents)
    if not resp.get('IsTruncated'):
        break
    kwargs['ContinuationToken'] = resp['NextContinuationToken']

print('Total objects found:', len(objects))

keys = [obj['Key'] for obj in objects if obj['Key'].endswith('.parquet')]
part_dates = []
for key in keys:
    for part in key.split('/'):
        if part.startswith('date='):
            part_dates.append(part.split('=', 1)[1])
            break

part_dates = sorted(set(part_dates))
print('Total partition dates:', len(part_dates))
print('First partition date:', part_dates[0] if part_dates else None)
print('Last partition date:', part_dates[-1] if part_dates else None)

start = datetime(2020, 2, 1).date()
end = datetime(2026, 2, 1).date()
expected = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
missing = [d for d in expected if d not in part_dates]
extra = [d for d in part_dates if d not in expected]
print('Expected dates count:', len(expected))
print('Missing dates count:', len(missing))
if missing:
    print('Missing first 20 dates:', missing[:20])
print('Extra dates count:', len(extra))
if extra:
    print('Extra first 20 dates:', extra[:20])

counts = Counter(part_dates)
print('Sample per-date parquet counts:')
for d in part_dates[:5]:
    print(' ', d, counts[d])
for d in part_dates[-5:]:
    print(' ', d, counts[d])

if missing:
    print('\nMissing range sample:')
    print(missing[:10])
    print(missing[-10:])
