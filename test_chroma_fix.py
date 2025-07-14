import sys
import pysqlite3

# Replace sqlite3 with pysqlite3 before importing chromadb
sys.modules['sqlite3'] = pysqlite3
print(f"Using SQLite version: {pysqlite3.sqlite_version}")

# Now import chromadb
import chromadb

# Fix for AttributeError: module 'chromadb' has no attribute 'config'
if not hasattr(chromadb, 'config'):
    chromadb.config = type('', (), {})()
if not hasattr(chromadb.config, 'Settings'):
    chromadb.config.Settings = chromadb.Settings

# Test if the fix works
try:
    settings = chromadb.config.Settings(is_persistent=True)
    settings.persist_directory = "./test_chroma"
    print("Successfully created chromadb.config.Settings")
    print("Fix is working correctly!")
except Exception as e:
    print(f"Error: {e}")
    print("Fix is not working correctly.")
