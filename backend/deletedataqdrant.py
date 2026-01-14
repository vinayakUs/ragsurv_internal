from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from backend.services.qdrant_service import get_qdrant_client
# Connect to your local Qdrant instance
client = get_qdrant_client()

collection_name = "knowledge_base"

# Delete all points (empty filter = delete all)
client.delete(
    collection_name=collection_name,
    points_selector=Filter(must=[])  # Correct way to specify "delete all"
)

print(f"✅ All data deleted from collection '{collection_name}', but the collection itself is preserved.")
        