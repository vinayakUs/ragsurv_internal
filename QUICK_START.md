# Quick Start Guide - Enhanced RAG System

## 🚀 Quick Start (3 Steps)

### Step 1: Rebuild BM25 Index
```bash
cd "D:\knowledge base"
python rebuild_bm25_index.py
```

### Step 2: Start Backend (if not running)
```bash
python start_backend.py
```

### Step 3: Test Queries
```bash
# Test a single query
python test_query.py --query "What is ESM?"

# Run full test suite
python test_query.py --suite
```

## 📋 What Changed?

Your RAG system now uses **Enhanced Hybrid Search**:

1. **Retrieves top-5** from both BM25 (keyword) and Vector (semantic) search
2. **Re-ranks** by combining scores (60% BM25, 40% Vector)
3. **Selects top-3** most relevant chunks
4. **Generates precise answers** using only the best matches

## ✅ Benefits

- ✅ **More accurate answers** - Only uses the 3 most relevant chunks
- ✅ **Better matching** - Weighted scoring favors exact keyword matches
- ✅ **Consistent results** - Temperature set to 0.0 for deterministic outputs
- ✅ **Traceable sources** - Shows which documents were used
- ✅ **Confidence scores** - Know how reliable each answer is

## 🔍 Testing Your System

### Test via Script:
```bash
python test_query.py --query "your question here"
```

### Test via API:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is ESM?"}'
```

### Test via Frontend:
Open `http://localhost:4200` and use the chat interface.

## 🔧 Maintenance

### When to Rebuild BM25 Index:
- After adding new documents to `D:\knowledge base\Document for test`
- After modifying existing documents
- If search results seem outdated
- Automatically happens on backend startup (for new docs)

### Command:
```bash
python rebuild_bm25_index.py
```

## 📊 Understanding Results

### High Confidence (≥0.85):
- Strong keyword + semantic match
- Answer likely very accurate
- Multiple sources agree

### Medium Confidence (0.50-0.84):
- Moderate keyword or semantic match
- Answer is reasonable but verify if critical
- Sources partially match query

### Low Confidence (<0.50):
- Weak match in knowledge base
- Consider rephrasing query or adding documents
- May need human verification

## 🎯 Best Practices

### Writing Effective Queries:
- ✅ Use specific keywords from your documents
- ✅ Ask focused questions
- ✅ Include relevant terms (names, dates, concepts)
- ❌ Avoid overly broad questions
- ❌ Don't use jargon not in your documents

### Example Good Queries:
- "What is ESM?"
- "When was the sprint planning session?"
- "Who attended the meeting?"
- "What are the system requirements?"

### Example Poor Queries:
- "Tell me everything"
- "What's happening?"
- "Information please"

## 🛠️ Troubleshooting

### Problem: "No documents found"
**Solution**: Run `python rebuild_bm25_index.py`

### Problem: Low quality answers
**Solution**: 
1. Check if documents are processed: `GET http://localhost:8000/documents`
2. Rebuild BM25 index
3. Verify documents contain relevant information

### Problem: Wrong sources cited
**Solution**: 
1. Clear and rebuild indexes
2. Check document content quality
3. Rephrase query with more specific terms

### Problem: "BM25 index loaded with 0 documents"
**Solution**: Run `python rebuild_bm25_index.py`

## 📁 Key Files

- `backend/services/hybrid_search.py` - Re-ranking logic
- `backend/services/query_engine.py` - Query pipeline
- `rebuild_bm25_index.py` - Index rebuild utility
- `test_query.py` - Testing tool
- `data/bm25_index/bm25_index.pkl` - BM25 index file

## 🎓 Advanced Tuning

### Adjust BM25/Vector Weights:
Edit `backend/services/hybrid_search.py` line 90-91:
```python
BM25_WEIGHT = 0.6  # Increase for more keyword matching
VEC_WEIGHT = 0.4   # Increase for more semantic matching
```

### Change Top-K Selection:
Edit `backend/services/hybrid_search.py` line 99:
```python
top_3_chunks = merged_list[:3]  # Change 3 to 4 or 5
```

### Adjust LLM Temperature:
Edit `backend/services/query_engine.py` line 112:
```python
temperature=0.0  # Increase to 0.1 or 0.2 for more variation
```

## 📞 Support

If issues persist:
1. Check logs in console output
2. Verify Qdrant is running (`http://localhost:6333`)
3. Verify Ollama is running (`http://localhost:11434`)
4. Check document processing in `backend/main.py` logs

---

**Happy Querying! 🎉**
