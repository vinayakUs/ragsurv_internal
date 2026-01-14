import qdrant_client
from qdrant_client.http import models

print("Testing query_points...")
try:
    client = qdrant_client.QdrantClient(host="localhost", port=6333)
    cols = client.get_collections()
    print(f"Collections: {cols}")
    
    col_name = "knowledge_base"
    # Create dummy vector
    vector = [0.1] * 1024
    
    try:
        results = client.query_points(
            collection_name=col_name,
            query=vector,
            limit=1,
            with_payload=True
        )
        print(f"Results type: {type(results)}")
        print(f"Results: {results}")
    except Exception as e:
        print(f"query_points failed: {e}")
        
    # Try query method too
    try:
        results = client.query(
            collection_name=col_name,
            query_vector=vector,
            limit=1
        )
        print(f"query method results: {results}")
    except Exception as e:
        print(f"query method failed: {e}")

except Exception as e:
    print(f"Error: {e}")
