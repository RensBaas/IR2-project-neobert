import json
from pathlib import Path

import numpy as np
from scipy import stats


def cohens_d(x, y):
    """Cohen's d effect size"""
    nx, ny = len(x), len(y)
    mean_diff = np.mean(x) - np.mean(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2))
    return mean_diff / pooled_std


def load_query_scores(filepath):
    """Load query-level scores from JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)


def paired_t_test_dataset(bert_file, neobert_file, metric, dataset_name):
    """
    Perform paired t-test on a single dataset

    Args:
        bert_file: Path to BERT query scores JSON
        neobert_file: Path to NeoBERT query scores JSON
        metric: Metric to compare - ndcg@10
        dataset_name
    """

    # Load scores
    bert_scores = load_query_scores(bert_file)
    neobert_scores = load_query_scores(neobert_file)

    # Extract metric values for matching queries
    bert_values = []
    neobert_values = []

    for qid in bert_scores:
        if qid in neobert_scores:
            bert_values.append(bert_scores[qid][metric])
            neobert_values.append(neobert_scores[qid][metric])

    # Convert to numpy arrays
    bert_values = np.array(bert_values)
    neobert_values = np.array(neobert_values)

    # Calculate statistics
    n_queries = len(bert_values)
    bert_mean = np.mean(bert_values)
    neobert_mean = np.mean(neobert_values)
    improvement = ((neobert_mean - bert_mean) / bert_mean) * 100

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(neobert_values, bert_values)

    # Effect size
    effect_size = cohens_d(neobert_values, bert_values)

    # Interpret effect size
    if abs(effect_size) < 0.2:
        effect_interp = "negligible"
    elif abs(effect_size) < 0.5:
        effect_interp = "small"
    elif abs(effect_size) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    # Significance
    if p_value < 0.001:
        sig_str = "***"
        sig_interp = "highly significant"
    elif p_value < 0.01:
        sig_str = "**"
        sig_interp = "very significant"
    elif p_value < 0.05:
        sig_str = "*"
        sig_interp = "significant"
    else:
        sig_str = "ns"
        sig_interp = "not significant"

    return {
        "dataset": dataset_name,
        "metric": metric,
        "n_queries": n_queries,
        "bert_mean": bert_mean,
        "neobert_mean": neobert_mean,
        "improvement_pct": improvement,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": effect_size,
        "effect_interpretation": effect_interp,
        "significance": sig_str,
        "sig_interpretation": sig_interp,
    }


def main():
    results_dir = Path("/home/scur1736/results")

    # Define datasets and their metrics
    datasets_config = {
        "msmarco": {
            "metric": "MRR@10",
            "bert_file": "msmarco_bert_cls_query_scores.json",
            "neobert_file": "msmarco_neobert_cls_query_scores.json",
        }
    }

    # datasets (all use NDCG@10)
    datasets_eval = [
        # BEIR
        "trec-covid",
        "nfcorpus",
        "nq",
        "hotpotqa",
        "fiqa",
        "arguana",
        "webis-touche2020",
        "quora",
        "dbpedia-entity",
        "scidocs",
        "fever",
        "climate-fever",
        "scifact",
        # BRIGHT
        "biology",
        "earth_science",
        "economics",
        "psychology",
        "robotics",
        "stackoverflow",
        "sustainable_living",
        "leetcode",
        "pony",
        "aops",
        "theoremqa_questions",
        "theoremqa_theorems",
    ]

    for dataset in datasets_eval:
        datasets_config[dataset] = {
            "metric": "ndcg_cut_10",
            "bert_file": "bert_query_scores.json",
            "neobert_file": "neobert_query_scores.json",
        }

    # Run tests for all datasets
    all_results = []

    print("=" * 100)
    print("STATISTICAL SIGNIFICANCE TESTING: BERT vs NeoBERT")
    print("=" * 100)
    print()

    for dataset_name, config in datasets_config.items():
        if dataset_name == "msmarco":
            bert_path = results_dir / config["bert_file"]
            neobert_path = results_dir / config["neobert_file"]
        else:
            dataset_path = results_dir / dataset_name
            bert_path = dataset_path / config["bert_file"]
            neobert_path = dataset_path / config["neobert_file"]

        # Check if files exist
        if not bert_path.exists() or not neobert_path.exists():
            print(f"⚠ Skipping {dataset_name}: Files not found")
            continue

        print(f"\n{'─' * 100}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'─' * 100}")

        try:
            result = paired_t_test_dataset(bert_path, neobert_path, config["metric"], dataset_name)
            all_results.append(result)

            # Print detailed results
            print(f"  Metric:          {result['metric']}")
            print(f"  N queries:       {result['n_queries']}")
            print(f"  BERT mean:       {result['bert_mean']:.4f}")
            print(f"  NeoBERT mean:    {result['neobert_mean']:.4f}")
            print(f"  Improvement:     {result['improvement_pct']:+.2f}%")
            print(f"  t-statistic:     {result['t_statistic']:.4f}")
            print(f"  p-value:         {result['p_value']:.6f} {result['significance']}")
            print(f"  Significance:    {result['sig_interpretation']}")
            print(f"  Cohen's d:       {result['cohens_d']:.4f} ({result['effect_interpretation']} effect)")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(
        f"\n{'Dataset':<20} {'Metric':<12} {'BERT':<10} {'NeoBERT':<10} {'Δ%':<8} {'p-value':<12} {'Sig':<6} {'Cohen d':<10}"
    )
    print("─" * 100)

    for r in all_results:
        print(
            f"{r['dataset']:<20} {r['metric']:<12} {r['bert_mean']:<10.4f} {r['neobert_mean']:<10.4f} "
            f"{r['improvement_pct']:>+7.2f}% {r['p_value']:<12.6f} {r['significance']:<6} {r['cohens_d']:<10.4f}"
        )

    # Overall statistics
    print("\n" + "=" * 100)
    print("OVERALL STATISTICS")
    print("=" * 100)

    sig_count = sum(1 for r in all_results if r["p_value"] < 0.05)
    highly_sig_count = sum(1 for r in all_results if r["p_value"] < 0.001)
    avg_improvement = np.mean([r["improvement_pct"] for r in all_results])

    print(f"\nTotal datasets tested:              {len(all_results)}")
    print(f"Significant improvements (p<0.05):  {sig_count} ({sig_count / len(all_results) * 100:.1f}%)")
    print(f"Highly significant (p<0.001):       {highly_sig_count} ({highly_sig_count / len(all_results) * 100:.1f}%)")
    print(f"Average improvement:                {avg_improvement:+.2f}%")

    # Save results to JSON
    output_file = results_dir / "statistical_tests_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 100)

    # Interpretation guide
    print("\nINTERPRETATION GUIDE:")
    print("  Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("  Effect sizes (Cohen's d): <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")
    print("=" * 100)


if __name__ == "__main__":
    main()
