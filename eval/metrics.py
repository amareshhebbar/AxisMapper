import re
from typing import List, Dict

def compute_metrics(results: List[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {
            "n": 0, "exact_match": "0.00%", "precision": "0.00%", "recall": "0.00%",
            "macro_f1": "0.00%", "format_compliance": "0.00%", "total_hallucinations": 0,
            "hallucination_rate": "0.00%", "over_coding_rate": "0.00%", "under_coding_rate": "0.00%"
        }

    exact_matches = []
    precisions    = []
    recalls       = []
    f1s           = []
    hallucinations = 0
    total_predicted_codes = 0
    format_ok     = 0
    over_count    = 0
    under_count   = 0

    invalid_format_pattern = re.compile(r"[a-z-A-Z]{4,}") 

    for r in results:
        exp  = r["expected"]    
        pred = r["predicted"]   
        total_predicted_codes += len(pred)

        exact_matches.append(exp == pred)

        if pred:
            p = len(exp & pred) / len(pred)
        else:
            p = 0.0
            
        rec = len(exp & pred) / len(exp) if exp else 1.0
        f1  = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0

        precisions.append(p)
        recalls.append(rec)
        f1s.append(f1)

        hallucinations += len(r["hallucinated"])

        raw = r["raw_output"].strip()
        has_prose = bool(invalid_format_pattern.search(raw))
        if not has_prose and raw:
            format_ok += 1

        if len(pred) > len(exp) + 2:
            over_count += 1
        if len(pred) < len(exp) - 1:
            under_count += 1

    avg_p    = sum(precisions) / n
    avg_r    = sum(recalls) / n
    macro_f1 = sum(f1s) / n

    h_rate = hallucinations / total_predicted_codes if total_predicted_codes > 0 else 0.0

    return {
        "n": n,
        "exact_match":        f"{sum(exact_matches)/n:.2%}",
        "precision":          f"{avg_p:.2%}",
        "recall":             f"{avg_r:.2%}",
        "macro_f1":           f"{macro_f1:.2%}",
        "format_compliance":  f"{format_ok/n:.2%}",
        "total_hallucinations": hallucinations,
        "hallucination_rate": f"{h_rate:.2%}",  
        "over_coding_rate":   f"{over_count/n:.2%}",
        "under_coding_rate":  f"{under_count/n:.2%}",
    }