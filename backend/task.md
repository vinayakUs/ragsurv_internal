# Task: Implement SOLID Retrieval System (New Folder)

- [ ] **Setup**
    - [x] Create `backend/new_retrieval_system` directory <!-- id: 0 -->
    - [ ] Define Core Interfaces (`Retriever`, `FusionStrategy`) in `interfaces.py` <!-- id: 1 -->

- [ ] **Implement Strategies**
    - [ ] Implement `WeightedFusion` (replicating `hybrid_search.py` logic: 0.6 BM25 + 0.4 Vector) <!-- id: 2 -->
    - [ ] Implement `RRFusion` (Reciprocal Rank Fusion) <!-- id: 3 -->

- [ ] **Implement Retrievers**
    - [ ] Implement `QdrantRetriever` in `retrievers.py` (replicating `qdrant_service.py` logic) <!-- id: 4 -->
    - [ ] Implement `BM25Retriever` in `retrievers.py` (replicating `bm25_service.py` logic) <!-- id: 5 -->
    - [ ] Implement `HybridRetriever` in `retrievers.py` (Composite pattern) <!-- id: 6 -->

- [ ] **Factory & Usage**
    - [ ] Implement `RetrieverFactory` in `factory.py` <!-- id: 7 -->
    - [ ] Create a demo script `demo_new_system.py` to verify it works without touching `main.py` <!-- id: 8 -->

- [ ] **Verification**
    - [ ] Verify `WeightedFusion` matches existing `hybrid_search` output logic <!-- id: 9 -->
    - [ ] Verify `QdrantRetriever` fetches correct chunks <!-- id: 10 -->
