from harness.trec_io import read_queries, read_qrels
from harness.metrics import evaluate_run
import itertools
import json
from submission import bm25, retrieve

def grid_search_bm25(corpus="data/nfcorpus/corpus.jsonl", queries_path="data/nfcorpus/queries_dev.tsv", qrels_path="data/nfcorpus/qrels_dev.txt", k1_vals=[0.8,1.2,1.6], b_vals=[0.5,0.75,1.0]):
    retrieve.build_index(corpus, "index_dir_tmp")
    retrieve.load_index("index_dir_tmp")  # populates bm25._INDEX, bm25._IDF via bm25.build()

    queries = read_queries(queries_path)
    qrels = read_qrels(qrels_path)
    results = []

    for k1, b in itertools.product(k1_vals, b_vals):
        run = {qid: bm25.score(text, 10, k1=k1, b=b) for qid, text in queries}
        agg = evaluate_run(run, qrels, k=10)["aggregate"]
        entry = {"experiment": f"bm25_k1={k1}_b={b}", "k1": k1, "b": b, **agg}
        results.append(entry)
        print(f"k1={k1:.2f} b={b:.2f}  nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}")

    with open("dev/results/history.jsonl", "a") as f:
        for e in results:
            f.write(json.dumps(e) + "\n")
    return results

if __name__=="__main__":
    grid_search_bm25()