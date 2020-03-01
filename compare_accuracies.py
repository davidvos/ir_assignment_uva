import os
import json
import pickle as pkl
import operator
from pprint import pprint
import argparse

model_names = [
    "lsi_bow",
    "lsi_tfidf",
    "doc2vec",
    "doc2vec_vocab_size",
    "doc2vec_window_size",
    "doc2vec_vec_dim",
    "word2vec",
    "lsa_bow"
]

best_run_results = {
    "tfidf":"./results/tfidf.json",
    "word2vec": "./results/skip_gram.json",
    "doc2vec":"./results/doc2vec_vocab_size_50000_results_trec.json",
    "lsi_tfidf":"./results/lsi-tfidf-embedding-2000-topics.json",
    "lsi_bow":"./results/lsi-bow-embedding-2000-topics.json",
    "lda_bow":"./results/lda-500-topics.json"
}

default_run_results = {
    "tfidf":"./results/tfidf.json",
    "word2vec": "./results/skip_gram.json",
    "doc2vec":"./results/doc2vec_vocab_size_10000_results_trec.json",
    "lsi_tfidf":"./results/lsi-tfidf-embedding-500-topics.json",
    "lsi_bow":"./results/lsi-bow-embedding-500-topics.json",
    "lda_bow":"./results/lda-500-topics.json"
}

lsi_bow_results = [
    "./results/lsi-bow-embedding-10-topics.json",
    "./results/lsi-bow-embedding-50-topics.json",
    "./results/lsi-bow-embedding-100-topics.json",
    "./results/lsi-bow-embedding-500-topics.json",
    "./results/lsi-bow-embedding-1000-topics.json",
    "./results/lsi-bow-embedding-2000-topics.json",
]

lsi_tfidf_results = [
    "./results/lsi-tfidf-embedding-10-topics.json",
    "./results/lsi-tfidf-embedding-50-topics.json",
    "./results/lsi-tfidf-embedding-100-topics.json",
    "./results/lsi-tfidf-embedding-500-topics.json",
    "./results/lsi-tfidf-embedding-1000-topics.json",
    "./results/lsi-tfidf-embedding-2000-topics.json",
]

doc2vec_window_size_results = [
    "./results/doc2vec_window_size_5_results_trec.json",
    "./results/doc2vec_window_size_5_results_trec.json",
    "./results/doc2vec_window_size_5_results_trec.json",
    "./results/doc2vec_window_size_5_results_trec.json"
]

doc2vec_vec_dim_results = [
    "./results/doc2vec_vec_dim_200_results_trec.json",
    "./results/doc2vec_vec_dim_300_results_trec.json",
    "./results/doc2vec_vec_dim_400_results_trec.json",
    "./results/doc2vec_vec_dim_500_results_trec.json"
]

doc2vec_vocab_size_results = [
    "./results/doc2vec_vocab_size_10000_results_trec.json",
    "./results/doc2vec_vocab_size_25000_results_trec.json",
    "./results/doc2vec_vocab_size_50000_results_trec.json",
    "./results/doc2vec_vocab_size_100000_results_trec.json",
    "./results/doc2vec_vocab_size_200000_results_trec.json"
]

doc2vec_results = doc2vec_window_size_results + doc2vec_vec_dim_results + doc2vec_vocab_size_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-model", type=str, help="Model to evaluate, determine best setup.")
    args = parser.parse_args()

    # Checks the best setup for a given model.
    if args.model:
        if not args.model in model_names:
            raise ValueError("Model should be specified as one of the following: {}".format(model_names))

        # query ids range from 51 to 200. evaluation set is 76 - 100, we should only use that for
        # parameter tuning.
        query_id_range = list(str(qid) for qid in range(76, 101))

        results_per_setup = {}
        for fn in eval(args.model + "_results"):
            with open(fn, "r") as f:
                all_results = json.load(f)

            results_per_setup[fn] = sum([all_results[qid]["map"] for qid in query_id_range]) / len(query_id_range)
            #results_per_setup[fn] = all_results["all"]["map"]
        
        setup, mean_eval_map = max(results_per_setup.items(), key=operator.itemgetter(1))

        print("######### TO DETERMINE THE BEST SETUP ###########\n\n")
        print("ALL RESULTS:")
        pprint(results_per_setup)
        print("\nBEST RESULT:\nnWith map of: {0:.4f} best setup: {1}".format(mean_eval_map, setup))

    else:
        
        print("\n\n############# AQ4.1 #############\n\n")

        all_results_per_setup = {}
        results_per_setup = {}
        eval_results_per_setup = {}
        for model, fn in best_run_results.items():
            with open(fn, "r") as f:
                res = json.load(f)
                res.pop("all", None)
                all_results_per_setup[model] = res

            results_per_setup[model] = {}
            results_per_setup[model]["map"] = sum([q["map"] for q in res.values()]) / len(res)
            results_per_setup[model]["ndcg"] = sum([q["ndcg"] for q in res.values()]) / len(res)

            # query ids range from 51 to 200. evaluation set is 76 - 100, we should only use that for
            # parameter tuning.
            query_id_range = list(str(qid) for qid in range(76, 101)) 
            eval_results_per_setup[model] = {}
            eval_results_per_setup[model]["map"] = sum([res[qid]["map"] for qid in query_id_range]) / len(query_id_range)
            eval_results_per_setup[model]["ndcg"] = sum([res[qid]["ndcg"] for qid in query_id_range]) / len(query_id_range)

        default_all_results_per_setup = {}
        default_results_per_setup = {}
        default_eval_results_per_setup = {}
        for model, fn in default_run_results.items():
            with open(fn, "r") as f:
                res = json.load(f)
                res.pop("all", None)
                default_all_results_per_setup[model] = res

            default_results_per_setup[model] = {}
            default_results_per_setup[model]["map"] = sum([q["map"] for q in res.values()]) / len(res)
            default_results_per_setup[model]["ndcg"] = sum([q["ndcg"] for q in res.values()]) / len(res)

            # query ids range from 51 to 200. evaluation set is 76 - 100, we should only use that for
            # parameter tuning.
            query_id_range = list(str(qid) for qid in range(76, 101)) 
            default_eval_results_per_setup[model] = {}
            default_eval_results_per_setup[model]["map"] = sum([res[qid]["map"] for qid in query_id_range]) / len(query_id_range)
            default_eval_results_per_setup[model]["ndcg"] = sum([res[qid]["ndcg"] for qid in query_id_range]) / len(query_id_range)        

        print("RETRIEVAL PERFORMANCE ON ALL QUERIES:")
        pprint(results_per_setup)

        print("\nRETRIEVAL PERFORMANCE ON EVAL QUERIES 76-100")
        pprint(eval_results_per_setup)

        print("\n\n############# AQ4.2 ##############\n\n")

        for model, res in all_results_per_setup.items():

            comp_with = dict(all_results_per_setup)
            comp_with.pop(model)
            for comp_model, comp_res in comp_with.items():
                pass

