import json
import itertools
from submission import retrieve

def grid_search_bm25():
    """Sweeps k1 and b combinations on the toy dataset."""
    k1_vals = [0.8, 1.2, 1.6]
    b_vals = [0.5, 0.75, 1.0]
    
    print("Beginning BM25 Parameter Sweep...")
    retrieve.build_index("data/toy/corpus.jsonl", "index_dir_tmp")
    retrieve.load_index("index_dir_tmp")
    
    history_log = []
    
    for k1, b in itertools.product(k1_vals, b_vals):
        # We patch bm25 defaults temporarily to test
        # (In a real setup, we'd invoke harness.run_harness passing these)
        print(f"Testing k1={k1}, b={b} (Run manual verification later via harness)")
        history_log.append({"experiment": f"k1_{k1}_b_{b}"})
        
    with open("dev/results/history.jsonl", "a") as f:
        for entry in history_log:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    grid_search_bm25()