from src.preprocess import load_data, prepare_data
from src.model import train_sentiment_model, save_model
from src.clustering import run_clustering, get_top_words_per_cluster


def main():
    file_path = "data/Amazon_Reviews.csv"

    print("Loading data...")
    df = load_data(file_path)

    print("Preparing data...")
    df = prepare_data(df)

    print("Training model...")
    model, vectorizer, results = train_sentiment_model(df)

    print("\nAccuracy:", results["accuracy"])
    print("\nClassification Report:")
    print(results["report"])

    print("\nSaving model...")
    save_model(
        model,
        vectorizer,
        "outputs/results/model.pkl",
        "outputs/results/vectorizer.pkl"
    )

    print("\nRunning clustering...")
    negative_reviews, vec, kmeans = run_clustering(df)

    cluster_words = get_top_words_per_cluster(vec, kmeans)

    print("\nTop words per cluster:")
    for k, v in cluster_words.items():
        print(f"Cluster {k}: {v}")

    negative_reviews.to_csv("outputs/results/negative_reviews.csv", index=False)
    df.to_csv("outputs/results/cleaned_data.csv", index=False)

    print("\nDone! Files saved in outputs/results/")


if __name__ == "__main__":
    main()