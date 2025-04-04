import numpy as np
import polars as pl


def create_popularity_baseline(train_df, test_df):
    """
    Create a popularity-based baseline model for MIND dataset using Polars

    Args:
        train_df: Polars DataFrame with training data
        test_df: Polars DataFrame with test data

    Returns:
        Polars DataFrame with predictions for test set
    """
    # Extract clicked articles from training set
    # First, explode impressions
    train_impressions = (
        train_df.select(pl.col("impressions"))
        .with_columns(pl.col("impressions").str.split(" "))
        .explode("impressions")
    )

    # Then parse news_id and click
    article_clicks = (
        train_impressions.with_columns(
            [
                pl.col("impressions").str.split("-").list.get(0).alias("news_id"),
                pl.col("impressions").str.split("-").list.get(1).alias("click"),
            ]
        )
        .filter(pl.col("click") == "1")
        .group_by("news_id")
        .len()
        .rename({"len": "clicks"})
    )

    # Calculate global CTR
    impressions_count = train_impressions.shape[0]
    clicks_count = train_impressions.filter(
        pl.col("impressions").str.contains("-1$")
    ).shape[0]
    global_ctr = clicks_count / impressions_count if impressions_count > 0 else 0.1

    # Prepare test data
    test_exploded = (
        test_df.select(pl.col("impression_id"), pl.col("impressions"))
        .with_columns(pl.col("impressions").str.split(" "))
        .explode("impressions")
        .with_columns(
            [pl.col("impressions").str.split("-").list.get(0).alias("news_id")]
        )
    )

    # Generate random noise
    np.random.seed(42)  # For reproducibility
    test_exploded = test_exploded.with_columns(
        pl.lit(np.random.random(test_exploded.shape[0]) * 0.001).alias("noise")
    )

    # Join with article clicks
    predictions = (
        test_exploded.join(article_clicks, on="news_id", how="left")
        .with_columns(
            [
                pl.when(pl.col("clicks").is_null())
                .then(global_ctr + pl.col("noise"))
                .otherwise(pl.col("clicks") + pl.col("noise"))
                .alias("score")
            ]
        )
        .select("impression_id", "news_id", "score")
    )

    return predictions


def normalize_scores(predictions_df):
    """Normalize scores to [0,1] range within each impression using Polars"""
    df = predictions_df

    # Group by impression_id and compute min and max scores
    result = df.join(
        df.group_by("impression_id").agg(
            [
                pl.col("score").min().alias("min_score"),
                pl.col("score").max().alias("max_score"),
            ]
        ),
        on="impression_id",
    )

    # Apply min-max normalization
    result = result.with_columns(
        [
            pl.when(pl.col("max_score") > pl.col("min_score"))
            .then(
                (pl.col("score") - pl.col("min_score"))
                / (pl.col("max_score") - pl.col("min_score"))
            )
            .otherwise(pl.lit(0.5))
            .alias("score")
        ]
    )

    # Drop temporary columns and select original columns
    result = result.drop(["min_score", "max_score"])

    return result
