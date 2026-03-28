import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_sentiment_model(df):
    """Train sentiment model."""
    X = df["clean_text"]
    y = df["sentiment"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred)
    }

    return model, vectorizer, results


def save_model(model, vectorizer, model_path, vectorizer_path):
    """Save trained model."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)