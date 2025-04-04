import json
import pandas as pd
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
from scipy.sparse import lil_matrix
from IPython.display import display


def extract_entities(entity_string):
    if entity_string is None or pd.isna(entity_string):
        return []

    try:
        entities = json.loads(entity_string)
        return [entity["Label"] for entity in entities]
    except:
        return []


def create_content_features(news_df):
    # First check if article_id exists, otherwise rename it from id if needed
    if "id" in news_df.columns and "article_id" not in news_df.columns:
        news_df = news_df.rename(columns={"id": "article_id"})

    # Create text_content column by combining title and abstract
    news_df = news_df.with_columns(
        [
            pl.concat_str(
                [
                    pl.col("title").fill_null(""),
                    pl.lit(" "),
                    pl.col("abstract").fill_null(""),
                ]
            ).alias("text_content")
        ]
    )

    # Add category information to create content_with_categories
    news_df = news_df.with_columns(
        [
            pl.concat_str(
                [
                    pl.col("category").fill_null(""),
                    pl.lit(" "),
                    pl.col("subcategory").fill_null(""),
                    pl.lit(" "),
                    pl.col("text_content"),
                ]
            ).alias("content_with_categories")
        ]
    )

    # Function to extract entity labels from JSON string
    def extract_entity_labels(json_str):
        if json_str is None or json_str == "" or json_str == "[]":
            return ""

        try:
            # Parse the JSON string
            entities = eval(json_str)
            # Extract the "Label" field from each entity
            labels = [
                entity.get("Label", "")
                for entity in entities
                if isinstance(entity, dict)
            ]
            # Join the labels with spaces
            return " ".join(labels)
        except:
            return ""

    # Process entities using the extraction function
    news_df = news_df.with_columns(
        [
            pl.col("title_entities")
            .map_elements(extract_entity_labels, return_dtype=pl.Utf8)
            .alias("title_entity_text"),
            pl.col("abstract_entities")
            .map_elements(extract_entity_labels, return_dtype=pl.Utf8)
            .alias("abstract_entity_text"),
        ]
    )

    # Combine entity texts
    news_df = news_df.with_columns(
        [
            pl.concat_str(
                [
                    pl.col("title_entity_text"),
                    pl.lit(" "),
                    pl.col("abstract_entity_text"),
                ]
            ).alias("entity_text")
        ]
    )

    # Create final content_features column
    news_df = news_df.with_columns(
        [
            pl.concat_str(
                [pl.col("content_with_categories"), pl.lit(" "), pl.col("entity_text")]
            ).alias("content_features")
        ]
    )

    display(news_df)

    return news_df


def vectorize_content(news_df):
    # Make sure we use the correct ID column name
    id_col = "article_id" if "article_id" in news_df.columns else "id"

    # Keep only the needed columns and ensure no null values in content_features
    content_df = news_df.select(
        [pl.col(id_col), pl.col("content_features").fill_null("")]
    )

    # Extract content to a list for vectorization
    content_features_list = content_df.get_column("content_features").to_list()
    ids_list = content_df.get_column(id_col).to_list()

    # Initialize and fit the TF-IDF vectorizer with better parameters
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.85,  # Ignore terms that appear in more than 85% of documents
        ngram_range=(1, 2),  # Include both unigrams and bigrams
    )

    # Fit and transform the content
    content_vectors = vectorizer.fit_transform(content_features_list)

    # Create mappings using Polars-extracted data
    id_to_idx = {news_id: idx for idx, news_id in enumerate(ids_list)}
    idx_to_id = {idx: news_id for news_id, idx in id_to_idx.items()}

    return content_vectors, vectorizer, id_to_idx, idx_to_id


# Build similarity matrix
def build_similarity_matrix_in_chunks(content_vectors, chunk_size=1000):
    """
    Build similarity matrix in chunks to avoid memory issues.

    Args:
        content_vectors: Sparse matrix of TF-IDF vectors
        chunk_size: Number of vectors to process at once

    Returns:
        scipy.sparse.csr_matrix: Sparse similarity matrix
    """
    n_samples = content_vectors.shape[0]
    similarity_matrix = lil_matrix((n_samples, n_samples))

    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        chunk_vectors = content_vectors[i:end_i]

        # Calculate similarity between current chunk and all vectors
        chunk_similarities = cosine_similarity(chunk_vectors, content_vectors)

        similarity_matrix[i:end_i] = chunk_similarities

        # Optional: Print progress
        print(f"Processed vectors {i} to {end_i-1} of {n_samples}")

    return similarity_matrix.tocsr()


# Get recommendations
def get_recommendations(news_id, news_df, similarity_matrix, id_to_idx, top_n=10):
    if news_id not in id_to_idx:
        return pl.DataFrame()

    idx = id_to_idx[news_id]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[
        1 : top_n + 1
    ]  # Skip the first item which is the input article

    # Get news indices
    news_indices = [i[0] for i in sim_scores]

    # Map back to news IDs
    idx_to_id = {idx: id for id, idx in id_to_idx.items()}
    recommended_ids = [idx_to_id[idx] for idx in news_indices]

    # Get the recommendation scores
    scores = [score for _, score in sim_scores]

    # Create a result dataframe
    result_df = pl.DataFrame({"id": recommended_ids, "score": scores})

    # Join with news_df to get more details if needed
    return result_df


# User-specific recommendations based on history
def get_user_recommendations(
    user_id, behaviors_df, news_df, similarity_matrix, id_to_idx, top_n=10
):
    # Find user's history
    user_rows = behaviors_df.filter(pl.col("user_id") == user_id)

    if len(user_rows) == 0:
        return get_popular_recommendations(news_df, top_n)

    user_history = user_rows.select("history").row(0)[0]

    if user_history is None or user_history == "":
        # If no history, return popular items
        return get_popular_recommendations(news_df, top_n)

    # Get history items
    history_items = user_history.split()

    # Get recommendations for each history item
    all_recommendations = []
    scores = {}

    for item in history_items:
        if item in id_to_idx:
            recs = get_recommendations(
                item, news_df, similarity_matrix, id_to_idx, top_n=20
            )

            for rec_id, score in zip(recs["id"], recs["score"]):
                if rec_id not in scores:
                    scores[rec_id] = 0
                scores[rec_id] += score  # Use similarity score for weighting

    # Sort by score
    sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_rec_ids = [item[0] for item in sorted_recs[:top_n]]
    top_rec_scores = [item[1] for item in sorted_recs[:top_n]]

    # Create recommendation dataframe
    result_df = pl.DataFrame({"id": top_rec_ids, "score": top_rec_scores})

    return result_df


# Get popular recommendations if no history
def get_popular_recommendations(news_df, top_n=10):
    # This would typically use click counts, but for simplicity we'll just return random items
    sample_ids = news_df.select("id").sample(n=top_n)["id"]

    # Create a fake score for each
    result_df = pl.DataFrame(
        {"id": sample_ids, "score": np.random.uniform(0.5, 1.0, len(sample_ids))}
    )

    return result_df


# Generate predictions for evaluation
def generate_predictions(behaviors_df, news_df, similarity_matrix, id_to_idx):
    """Generate predictions for all impressions in the behaviors_df"""

    # Process impressions
    impressions_df = (
        behaviors_df.select(
            pl.col("impression_id"), pl.col("user_id"), pl.col("impressions")
        )
        .with_columns(pl.col("impressions").str.split(" "))
        .explode("impressions")
        .with_columns(
            [
                pl.col("impressions").str.split("-").list.get(0).alias("news_id"),
            ]
        )
        .drop("impressions")
    )

    # Group by impression and user
    predictions = []

    # Process each impression
    for impression_group in tqdm(impressions_df.group_by(["impression_id", "user_id"])):
        impression_id = impression_group[0][0]
        user_id = impression_group[0][1]
        impression_df = impression_group[1]

        # Get candidate news IDs
        candidate_news_ids = impression_df["news_id"].to_list()

        # Get user history
        user_rows = behaviors_df.filter(pl.col("user_id") == user_id)

        if len(user_rows) == 0:
            continue

        user_history = user_rows.select("history").row(0)[0]

        if user_history is None or user_history == "":
            # If no history, use general popularity
            pred_scores = np.random.uniform(0.5, 1.0, len(candidate_news_ids))

            for news_id, score in zip(candidate_news_ids, pred_scores):
                predictions.append(
                    {"impression_id": impression_id, "news_id": news_id, "score": score}
                )
            continue

        # Get history items
        history_items = user_history.split()

        # Score each candidate news item
        for news_id in candidate_news_ids:
            score = 0.0
            count = 0

            for hist_item in history_items:
                if hist_item in id_to_idx and news_id in id_to_idx:
                    hist_idx = id_to_idx[hist_item]
                    news_idx = id_to_idx[news_id]
                    sim_score = similarity_matrix[hist_idx, news_idx]
                    score += sim_score
                    count += 1

            # Normalize score
            final_score = score / count if count > 0 else 0.5

            predictions.append(
                {
                    "impression_id": impression_id,
                    "news_id": news_id,
                    "score": final_score,
                }
            )

    return pl.DataFrame(predictions)
