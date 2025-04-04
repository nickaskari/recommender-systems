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
    # Better concatenation of text with explicit weights
    news_df = news_df.with_columns(
        [
            # Multiply category and subcategory by 3 to increase their importance
            pl.concat_str(
                [
                    # Triple the category by concatenating it with itself three times
                    pl.concat_str(
                        [
                            pl.col("category").fill_null(""),
                            pl.lit(" "),
                            pl.col("category").fill_null(""),
                            pl.lit(" "),
                            pl.col("category").fill_null(""),
                        ]
                    ),
                    pl.lit(" "),
                    # Triple the subcategory
                    pl.concat_str(
                        [
                            pl.col("subcategory").fill_null(""),
                            pl.lit(" "),
                            pl.col("subcategory").fill_null(""),
                            pl.lit(" "),
                            pl.col("subcategory").fill_null(""),
                        ]
                    ),
                    pl.lit(" "),
                    pl.col("title").fill_null(""),  # Title is naturally important
                    pl.lit(" "),
                    pl.col("abstract").fill_null(""),
                ]
            ).alias("content_with_categories")
        ]
    )

    # Extract and weight entity text (entities are highly valuable signals)
    news_df = news_df.with_columns(
        [
            pl.col("title_entities")
            .map_elements(
                lambda x: (
                    " ".join(
                        [
                            entity["Label"] + " " + entity["Label"]
                            for entity in json.loads(x)
                        ]
                    )
                    if x and x != "[]" and x != "null"
                    else ""
                ),
                return_dtype=pl.Utf8,
            )
            .alias("title_entity_text"),
            pl.col("abstract_entities")
            .map_elements(
                lambda x: (
                    " ".join([entity["Label"] for entity in json.loads(x)])
                    if x and x != "[]" and x != "null"
                    else ""
                ),
                return_dtype=pl.Utf8,
            )
            .alias("abstract_entity_text"),
        ]
    )

    # Create final content features
    news_df = news_df.with_columns(
        [
            pl.concat_str(
                [
                    pl.col("content_with_categories"),
                    pl.lit(" "),
                    # Double the title entities by concatenating with itself
                    pl.col("title_entity_text"),
                    pl.lit(" "),
                    pl.col("title_entity_text"),
                    pl.lit(" "),
                    pl.col("abstract_entity_text"),
                ]
            ).alias("content_features")
        ]
    )

    return news_df


def vectorize_content(news_df):
    id_col = "article_id" if "article_id" in news_df.columns else "id"

    # Clean content features first
    content_df = news_df.select(
        [
            pl.col(id_col),
            pl.col("content_features")
            .fill_null("")
            .str.replace(r"[^\w\s]", " ", literal=False),
        ]
    )

    content_features_list = content_df.get_column("content_features").to_list()
    ids_list = content_df.get_column(id_col).to_list()

    # Better TF-IDF parameters
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased from 5000
        min_df=3,  # Ignore terms in fewer than 3 documents
        max_df=0.8,  # Ignore terms in more than 80% of documents
        ngram_range=(1, 2),  # Include both unigrams and bigrams
        stop_words="english",
        sublinear_tf=True,  # Apply sublinear tf scaling (log scaling)
    )

    content_vectors = vectorizer.fit_transform(content_features_list)

    # Create mappings
    id_to_idx = {news_id: idx for idx, news_id in enumerate(ids_list)}

    return content_vectors, vectorizer, id_to_idx


# Build similarity matrix
def build_similarity_matrix(content_vectors, chunk_size=5000):
    n_samples = content_vectors.shape[0]
    similarity_matrix = lil_matrix((n_samples, n_samples))

    for i in tqdm(range(0, n_samples, chunk_size), desc="Building similarity matrix"):
        end_i = min(i + chunk_size, n_samples)
        chunk_vectors = content_vectors[i:end_i]

        # Calculate similarity between current chunk and all vectors
        chunk_similarities = cosine_similarity(chunk_vectors, content_vectors)

        similarity_matrix[i:end_i] = chunk_similarities

    return similarity_matrix.tocsr()  # Convert to CSR for efficient row slicing


def get_time_aware_user_recommendations(
    user_id, behaviors_df, news_df, similarity_matrix, id_to_idx, top_n=10
):
    # Find user's history with time information
    user_rows = behaviors_df.filter(pl.col("user_id") == user_id)

    if len(user_rows) == 0:
        return get_popular_recommendations(news_df, top_n)

    # Parse time information
    user_rows = user_rows.with_columns(
        [
            pl.col("time")
            .str.strptime(pl.Datetime, "%m/%d/%Y %I:%M:%S %p")
            .alias("datetime")
        ]
    )

    # Sort by time to weight recent history more heavily
    user_rows = user_rows.sort("datetime", descending=True)

    # Get history from all user rows, with most recent first
    all_history = []
    for row in user_rows.iter_rows(named=True):
        if row["history"] and row["history"] != "":
            all_history.extend(row["history"].split())

    # Remove duplicates while preserving order (first occurrence is kept - most recent)
    history_items = []
    for item in all_history:
        if item not in history_items:
            history_items.append(item)

    if not history_items:
        return get_popular_recommendations(news_df, top_n)

    # Apply time decay - more weight to recent items
    scores = {}
    time_weights = [max(0.5, 1.0 - (0.1 * i)) for i in range(len(history_items))]

    for i, item in enumerate(history_items):
        if item in id_to_idx:
            recs = get_recommendations(
                item, news_df, similarity_matrix, id_to_idx, top_n=20
            )

            for rec_id, score in zip(recs["id"], recs["score"]):
                if rec_id not in scores:
                    scores[rec_id] = 0
                # Apply time decay weight
                scores[rec_id] += score * time_weights[i]

    # Filter out items in history
    scores = {k: v for k, v in scores.items() if k not in history_items}

    # Sort by score
    sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_rec_ids = [item[0] for item in sorted_recs[:top_n]]
    top_rec_scores = [item[1] for item in sorted_recs[:top_n]]

    # Create recommendation dataframe
    result_df = pl.DataFrame({"id": top_rec_ids, "score": top_rec_scores})

    return result_df


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


def get_diverse_recommendations(
    user_id, behaviors_df, news_df, similarity_matrix, id_to_idx, top_n=10
):
    # Get base recommendations
    base_recs = get_time_aware_user_recommendations(
        user_id, behaviors_df, news_df, similarity_matrix, id_to_idx, top_n=top_n * 3
    )

    if len(base_recs) == 0:
        return get_popular_recommendations(news_df, top_n)

    # Get user category preferences
    user_rows = behaviors_df.filter(pl.col("user_id") == user_id)

    if len(user_rows) == 0:
        return base_recs.head(top_n)

    user_history = user_rows.select("history").row(0)[0]

    if user_history is None or user_history == "":
        return base_recs.head(top_n)

    history_items = user_history.split()

    # Get categories of history items
    history_categories = []
    for item in history_items:
        cat_row = news_df.filter(pl.col("id") == item).select("category")
        if len(cat_row) > 0:
            history_categories.append(cat_row.row(0)[0])

    # Count categories
    category_counts = {}
    for cat in history_categories:
        if cat in category_counts:
            category_counts[cat] += 1
        else:
            category_counts[cat] = 1

    # Calculate category preference scores
    total_history = len(history_items)
    category_prefs = {
        cat: count / total_history for cat, count in category_counts.items()
    }

    # Get diverse recommendations (limit repetition of same category)
    rec_ids = base_recs["id"].to_list()
    rec_scores = base_recs["score"].to_list()

    # Get categories for recommendations
    rec_categories = []
    for item in rec_ids:
        cat_row = news_df.filter(pl.col("id") == item).select("category")
        if len(cat_row) > 0:
            rec_categories.append(cat_row.row(0)[0])
        else:
            rec_categories.append("unknown")

    # Apply diversity algorithm (Maximal Marginal Relevance simplified)
    selected_indices = []
    selected_categories = []

    # First select one item from each preferred category
    for cat in category_prefs:
        if len(selected_indices) >= top_n:
            break

        # Find best item from this category not already selected
        best_idx = -1
        best_score = -1

        for i, (item_id, score, category) in enumerate(
            zip(rec_ids, rec_scores, rec_categories)
        ):
            if i in selected_indices:
                continue

            if category == cat and score > best_score:
                best_score = score
                best_idx = i

        if best_idx != -1:
            selected_indices.append(best_idx)
            selected_categories.append(cat)

    # Fill remaining slots with highest scores, avoiding too many from same category
    remaining_spots = top_n - len(selected_indices)
    if remaining_spots > 0:
        category_counts = {}
        for cat in selected_categories:
            if cat in category_counts:
                category_counts[cat] += 1
            else:
                category_counts[cat] = 1

        # Sort remaining items by score
        remaining_items = [
            (i, score)
            for i, score in enumerate(rec_scores)
            if i not in selected_indices
        ]
        remaining_items.sort(key=lambda x: x[1], reverse=True)

        for i, score in remaining_items:
            if len(selected_indices) >= top_n:
                break

            category = rec_categories[i]
            count = category_counts.get(category, 0)

            # Limit to at most 3 articles from same category
            if count < 3:
                selected_indices.append(i)
                selected_categories.append(category)

                if category in category_counts:
                    category_counts[category] += 1
                else:
                    category_counts[category] = 1

    # Get final recommendations
    final_ids = [rec_ids[i] for i in selected_indices]
    final_scores = [rec_scores[i] for i in selected_indices]

    # Create final dataframe
    result_df = pl.DataFrame({"id": final_ids, "score": final_scores})

    return result_df


def generate_enhanced_predictions(behaviors_df, news_df, similarity_matrix, id_to_idx):
    """Generate predictions with improved scoring for all impressions"""

    # Process impressions
    impressions_df = (
        behaviors_df.select(
            pl.col("impression_id"),
            pl.col("user_id"),
            pl.col("time"),
            pl.col("impressions"),
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

    # Group by impression
    predictions = []

    # Process each impression
    for impression_group in tqdm(
        impressions_df.group_by(["impression_id", "user_id", "time"]),
        desc="Generating predictions",
    ):
        impression_id = impression_group[0][0]
        user_id = impression_group[0][1]
        timestamp = impression_group[0][2]
        impression_df = impression_group[1]

        # Get candidate news IDs
        candidate_news_ids = impression_df["news_id"].to_list()

        # Get user's history
        user_rows = behaviors_df.filter(pl.col("user_id") == user_id)

        if len(user_rows) == 0:
            # No history - use random scores
            for news_id in candidate_news_ids:
                predictions.append(
                    {
                        "impression_id": impression_id,
                        "news_id": news_id,
                        "score": np.random.uniform(0.4, 0.6),
                    }
                )
            continue

        user_history = user_rows.select("history").row(0)[0]

        if user_history is None or user_history == "":
            # Empty history - use random scores
            for news_id in candidate_news_ids:
                predictions.append(
                    {
                        "impression_id": impression_id,
                        "news_id": news_id,
                        "score": np.random.uniform(0.4, 0.6),
                    }
                )
            continue

        # Get history items
        history_items = user_history.split()

        # Get category preferences
        history_categories = []
        for item in history_items:
            cat_row = news_df.filter(pl.col("id") == item).select("category")
            if len(cat_row) > 0:
                history_categories.append(cat_row.row(0)[0])

        # Calculate category preferences
        category_counts = {}
        for cat in history_categories:
            if cat in category_counts:
                category_counts[cat] += 1
            else:
                category_counts[cat] = 1

        # Score each candidate news item
        for news_id in candidate_news_ids:
            base_score = 0.0
            count = 0

            # Get item category for boosting
            news_cat = ""
            cat_row = news_df.filter(pl.col("id") == news_id).select("category")
            if len(cat_row) > 0:
                news_cat = cat_row.row(0)[0]

            # Category boost factor
            cat_boost = 1.0
            if news_cat in category_counts:
                cat_boost = 1.0 + (
                    0.2 * (category_counts[news_cat] / len(history_items))
                )

            # Calculate similarity-based score
            for i, hist_item in enumerate(history_items):
                if hist_item in id_to_idx and news_id in id_to_idx:
                    hist_idx = id_to_idx[hist_item]
                    news_idx = id_to_idx[news_id]

                    # Apply time decay - more recent items have more weight
                    recency_weight = max(0.5, 1.0 - (0.05 * i))

                    sim_score = similarity_matrix[hist_idx, news_idx] * recency_weight
                    base_score += sim_score
                    count += 1

            # Calculate final score
            if count > 0:
                final_score = (base_score / count) * cat_boost
            else:
                final_score = 0.5  # Default score

            predictions.append(
                {
                    "impression_id": impression_id,
                    "news_id": news_id,
                    "score": final_score,
                }
            )

    return pl.DataFrame(predictions)


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
