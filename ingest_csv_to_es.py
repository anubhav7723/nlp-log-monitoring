import pandas as pd
from elasticsearch import Elasticsearch



ES_HOST = "http://localhost:9200"
INDEX_NAME = "ai-logs-index"


es = Elasticsearch(ES_HOST)

if not es.ping():
    raise ValueError("❌ Cannot connect to Elasticsearch")

print("✅ Connected to Elasticsearch")

CSV_FILE_PATH = "datasets\test.csv"
df = pd.read_csv(CSV_FILE_PATH)

print(f"✅ Loaded {len(df)} log records from CSV")

