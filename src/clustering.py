from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def run_clustering(df, n_clusters=5):
    """Cluster negative reviews."""
    negative_reviews = df[df["sentiment"] == "Negative"].copy()

    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
    X_neg = vectorizer.fit_transform(negative_reviews["clean_text"])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    negative_reviews["cluster"] = kmeans.fit_predict(X_neg)

    return negative_reviews, vectorizer, kmeans


def get_top_words_per_cluster(vectorizer, kmeans, n_words=10):
    """Get top words for each cluster."""
    terms = vectorizer.get_feature_names_out()
    cluster_words = {}

    for i in range(kmeans.n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-n_words:][::-1]
        cluster_words[i] = [terms[j] for j in top_indices]

    return cluster_words
