import json
import sys
import os
from pathlib import Path
from datetime import datetime


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

PROCESSED_DIR = Path("data/processed")
SYNTHETIC_DIR = Path("data/synthetic")

REAL_TRIPLES_PATH = PROCESSED_DIR / "kg_triples_ids.txt"
MAPPINGS_PATH = PROCESSED_DIR / "kg_mappings.json"


OUTPUT_JSON = Path("dashboard_data.json")

def main():
    print("[INFO] Updating Dashboard Data...")


    with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
        id_to_name = json.load(f)


    synthetic_files = sorted(SYNTHETIC_DIR.glob("generated_*.txt"))
    if not synthetic_files:
        print("[WARN] No generated data found.")
        return

    latest_file = synthetic_files[-1]
    synthetic_triples = []
    with open(latest_file, "r") as f:
        next(f) 
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                r, t, score = parts[0], parts[1], float(parts[2])
                synthetic_triples.append((r, t, score))

    real_pairs = set()
    with open(REAL_TRIPLES_PATH, "r") as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) == 3:
                real_pairs.add(f"{p[1]}|{p[2]}")


    novel_count = sum(1 for r, t, score in synthetic_triples if f"{r}|{t}" not in real_pairs)
    total = len(synthetic_triples)
    novelty_score = (novel_count / total) * 100 if total > 0 else 0
    uniqueness_score = (len(set([x[1] for x in synthetic_triples])) / total) * 100 if total > 0 else 0


    decoded_hypotheses = []
    if synthetic_triples:
        r, t, score = synthetic_triples[0] 
        tail_name = id_to_name.get(t, t)
        rel_name = r.replace("dblp:", "").replace("inYear", "Published in").replace("publishedIn", "Venue:").replace("wrote", "Authored Paper")
        
        decoded_hypotheses.append({
            "relation": rel_name,
            "entity": tail_name,
            "score": f"{score:.4f}",
            "novel": (f"{r}|{t}" not in real_pairs)
        })

    # 6. JSON Kaydet
    dashboard_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "stats": {
            "novelty": round(novelty_score, 2),
            "uniqueness": round(uniqueness_score, 2),
            "total_generated": total
        },
        "hypotheses": decoded_hypotheses
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"[SUCCESS] Dashboard JSON saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()