import pandas as pd
import polars as pl
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_mind_predictions(
    predictions_df,
    behaviors_df,
    metrics=["auc", "mrr", "ndcg@5", "ndcg@10"],
):
    """
    Evaluate MIND dataset predictions using Polars

    Args:
        predictions_df: DataFrame with columns [impression_id, news_id, score]
        behaviors_df: DataFrame with ground truth behaviors
        metrics: List of metrics to compute

    Returns:
        Dictionary of evaluation metrics
    """
    # Convert to Polars if needed
    if isinstance(predictions_df, pd.DataFrame):
        predictions_df = pl.from_pandas(predictions_df)
    if isinstance(behaviors_df, pd.DataFrame):
        behaviors_df = pl.from_pandas(behaviors_df)

    # Process impressions to get ground truth clicks
    truth_df = (
        behaviors_df.select(pl.col("impression_id"), pl.col("impressions"))
        .with_columns(pl.col("impressions").str.split(" "))
        .explode("impressions")
        .with_columns(
            [
                pl.col("impressions").str.split("-").list.get(0).alias("news_id"),
                pl.col("impressions")
                .str.split("-")
                .list.get(1)
                .cast(pl.Int8)
                .alias("clicked"),
            ]
        )
        .drop("impressions")
    )

    # Merge predictions with ground truth
    evaluation_df = truth_df.join(
        predictions_df, on=["impression_id", "news_id"], how="inner"
    )

    # If empty dataframe after merge, something's wrong
    if len(evaluation_df) == 0:
        return {
            "error": "No matching impression_id and news_id between predictions and ground truth"
        }

    results = {}

    # Calculate AUC
    if "auc" in metrics:
        # Convert to numpy arrays for sklearn
        clicked = evaluation_df.get_column("clicked").to_numpy()
        scores = evaluation_df.get_column("score").to_numpy()
        results["auc"] = roc_auc_score(clicked, scores)

    # Calculate MRR
    if "mrr" in metrics:
        results["mrr"] = calculate_mrr(evaluation_df)

    # Calculate nDCG metrics
    if "ndcg@5" in metrics:
        results["ndcg@5"] = calculate_ndcg(evaluation_df, k=5)

    if "ndcg@10" in metrics:
        results["ndcg@10"] = calculate_ndcg(evaluation_df, k=10)

    return results


def calculate_mrr(evaluation_df):
    """Calculate Mean Reciprocal Rank using Polars"""
    mrr_scores = []

    # Group by impression_id
    for impression_id, group in evaluation_df.group_by("impression_id"):
        # Sort by score in descending order
        sorted_group = group.sort("score", descending=True)

        # Get clicks
        clicks = sorted_group.get_column("clicked").to_numpy()

        # Find position of first click (1-indexed)
        click_positions = np.where(clicks == 1)[0]

        if len(click_positions) > 0:
            first_click_pos = click_positions[0] + 1  # +1 for 1-indexed position
            mrr_scores.append(1.0 / first_click_pos)
        else:
            mrr_scores.append(0.0)

    return np.mean(mrr_scores) if mrr_scores else 0.0


def calculate_ndcg(evaluation_df, k=10):
    """Calculate normalized Discounted Cumulative Gain"""
    ndcg_scores = []

    # Group by impression_id
    for impression_id, group in evaluation_df.group_by("impression_id"):
        # Sort by score in descending order
        sorted_group = group.sort("score", descending=True)

        # Get clicks (relevance) limited to top k
        relevance = sorted_group.get_column("clicked").to_numpy()[:k]

        if len(relevance) == 0:
            ndcg_scores.append(0.0)
            continue

        # If less than k items, pad with zeros
        if len(relevance) < k:
            relevance = np.pad(relevance, (0, k - len(relevance)))

        # Calculate DCG
        positions = np.arange(1, len(relevance) + 1)
        dcg = np.sum(relevance / np.log2(positions + 1))

        # Calculate ideal DCG
        ideal_relevance = np.sort(relevance)[::-1]
        idcg = np.sum(ideal_relevance / np.log2(positions + 1))

        # Calculate nDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0
