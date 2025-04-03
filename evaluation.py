import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_mind_predictions(
    predictions_df,
    behaviors_path="data/MINDlarge_train/behaviors.tsv",
    metrics=["auc", "mrr", "ndcg@5", "ndcg@10"],
):
    """
    Evaluate MIND dataset predictions using pandas

    Args:
        predictions_df: DataFrame with columns [impression_id, news_id, score]
        behaviors_path: Path to behaviors.tsv with ground truth
        metrics: List of metrics to compute

    Returns:
        Dictionary of evaluation metrics
    """
    # Read behaviors file to get ground truth
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "timestamp", "history", "impressions"],
    )

    # Process impressions to get ground truth clicks
    truth_data = []

    for _, row in behaviors_df.iterrows():
        imp_id = row["impression_id"]
        impressions = row["impressions"].split()

        for imp in impressions:
            news_id, click = imp.split("-")
            truth_data.append(
                {"impression_id": imp_id, "news_id": news_id, "clicked": int(click)}
            )

    truth_df = pd.DataFrame(truth_data)

    # Merge predictions with ground truth
    evaluation_df = pd.merge(
        truth_df, predictions_df, on=["impression_id", "news_id"], how="inner"
    )

    # If empty dataframe after merge, something's wrong
    if len(evaluation_df) == 0:
        return {
            "error": "No matching impression_id and news_id between predictions and ground truth"
        }

    results = {}

    # Calculate AUC
    if "auc" in metrics:
        results["auc"] = roc_auc_score(evaluation_df["clicked"], evaluation_df["score"])

    # Calculate MRR and nDCG metrics
    if any(m in metrics for m in ["mrr", "ndcg@5", "ndcg@10"]):
        # Group by impression_id
        grouped = evaluation_df.groupby("impression_id")

        mrr_scores = []
        ndcg5_scores = []
        ndcg10_scores = []

        for imp_id, group in grouped:
            # Sort by predicted scores (descending)
            sorted_group = group.sort_values("score", ascending=False)

            # For MRR calculation
            if "mrr" in metrics:
                # Find position of first clicked item
                clicked_items = sorted_group[sorted_group["clicked"] == 1]
                if not clicked_items.empty:
                    first_click_pos = (
                        sorted_group["clicked"].values.tolist().index(1) + 1
                    )
                    mrr_scores.append(1.0 / first_click_pos)
                else:
                    mrr_scores.append(0.0)

            # For nDCG calculation
            clicks = sorted_group["clicked"].values

            if "ndcg@5" in metrics:
                ndcg5_scores.append(calculate_ndcg(clicks, k=5))

            if "ndcg@10" in metrics:
                ndcg10_scores.append(calculate_ndcg(clicks, k=10))

        if "mrr" in metrics:
            results["mrr"] = np.mean(mrr_scores)

        if "ndcg@5" in metrics:
            results["ndcg@5"] = np.mean(ndcg5_scores)

        if "ndcg@10" in metrics:
            results["ndcg@10"] = np.mean(ndcg10_scores)

    return results


def calculate_ndcg(relevance_scores, k=10):
    """Calculate normalized Discounted Cumulative Gain"""
    # Get top k items
    relevance = np.array(relevance_scores)[:k]

    # Calculate DCG
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

    # Calculate ideal DCG
    ideal_relevance = np.sort(relevance)[::-1]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

    # Return nDCG
    return dcg / idcg if idcg > 0 else 0.0


# For your predictions DataFrame, use this format:
# predictions_df = pd.DataFrame({
#     'impression_id': ['1', '1', '1', ...],
#     'news_id': ['N78206', 'N26368', 'N7578', ...],
#     'score': [0.2, 0.5, 0.3, ...]
# })
