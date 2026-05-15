from app.retrieval.hybrid_search import hybrid_search

results = hybrid_search(
    "Hiring Java backend developer with communication skills",
    top_k=5
)

for i, r in enumerate(results, 1):
    print(f"\n{i}. {r['assessment_name']}")
    print(r['assessment_url'])