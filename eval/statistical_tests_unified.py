import json
import numpy as np
from scipy import stats

# ============================================================================================
# Interpreting significance and effect size
# Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant
# Effect sizes (Cohen's d): <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large
# ============================================================================================

def cohens_d(x, y):
    """Cohen's d effect size"""
    nx, ny = len(x), len(y)
    mean_diff = np.mean(x) - np.mean(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2))
    return mean_diff / pooled_std


def load_query_scores(filepath):
    """Load query-level scores from JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)['per_query_results']


def paired_t_test_dataset(model1, model2, model1_file, model2_file, metric, dataset_name):
    """
    Perform paired t-test on a single dataset

    Args:
        model1: Name of Model 1
        model2: Name of Model 2
        model1_file: Path to Model 1 per query scores
        neobert_file: Path to Model 2 per query scores
        metric: Metric to compare - ndcg@10
        dataset_name: Name of the dataset to compare models on
    """

    # Load scores
    model1_scores = load_query_scores(model1_file)
    model2_scores = load_query_scores(model2_file)

    # Extract metric values for matching queries
    values1 = []
    values2 = []

    for qid in model1_scores:
        if qid in model2_scores:
            values1.append(model1_scores[qid][metric])
            values2.append(model2_scores[qid][metric])

    # Convert to numpy arrays
    values1 = np.array(values1)
    values2 = np.array(values2)

    # Calculate statistics
    n_queries = len(values1)
    model1_mean = np.mean(values1)
    model2_mean = np.mean(values2)
    improvement = ((model2_mean - model1_mean) / model1_mean) * 100 # model 2 improvement over model 1

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values2, values1)

    # Effect size
    effect_size = cohens_d(values2, values1)

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
        "model1": model1,
        "model2": model2,
        "metric": metric,
        "n_queries": n_queries,
        f"{model1}_mean": model1_mean,
        f"{model2}_mean": model2_mean,
        "improvement_pct": improvement,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": effect_size,
        "effect_interpretation": effect_interp,
        "significance": sig_str,
        "sig_interpretation": sig_interp,
    }