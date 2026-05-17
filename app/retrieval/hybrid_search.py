from app.retrieval.bm25_search import search_bm25

def hybrid_search(query, top_k=5):
    return search_bm25(query, top_k=top_k)