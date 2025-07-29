from django.shortcuts import render
from pathlib import Path
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import query_jaccard, query_tfidf  # query(q: str, q_amount: int)

def query_view(request):
    results_tfidf, results_jaccard = [], []
    if request.method == 'GET' and 'q' in request.GET and 'q_amount' in request.GET:
        q = request.GET.get('q')
        try:
            q_amount = int(request.GET.get('q_amount'))
            results_tfidf = query_tfidf(q, q_amount)
            results_jaccard = query_jaccard(q, q_amount)
        except ValueError:
            results = [{"desc": "Invalid input", "similarity": "N/A"}]

    return render(request,
                  'QueryApp/result.html',
                  {'results_tfidf': results_tfidf, 'results_jaccard': results_jaccard})