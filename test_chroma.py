import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

# Now import chromadb after replacing sqlite3
import chromadb

# Check SQLite version
print(f"SQLite version: {pysqlite3.sqlite_version}")

# Try to create a Chroma client
client = chromadb.Client()
print("Chroma client initialized successfully")