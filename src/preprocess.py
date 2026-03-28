import pandas as pd
import re
import string


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset safely."""
    df = pd.read_csv(
        file_path,
        encoding="latin1",
        engine="python",
        on_bad_lines="skip"
    )
    return df


def extract_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric rating from text."""
    df = df.copy()

    df["rating_num"] = df["Rating"].astype(str).str.extract(r"(\d)", expand=False)
    df["rating_num"] = pd.to_numeric(df["rating_num"], errors="coerce")

    df = df.dropna(subset=["rating_num"]).copy()
    df["rating_num"] = df["rating_num"].astype(int)

    return df


def create_sentiment_label(rating: int) -> str:
    """Convert rating into sentiment."""
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


def clean_text(text: str) -> str:
    """Clean text for NLP."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = df[["Country", "Review Date", "Rating", "Review Title", "Review Text"]].copy()

    df["Review Title"] = df["Review Title"].fillna("")
    df["Review Text"] = df["Review Text"].fillna("")

    df = extract_rating(df)

    df["sentiment"] = df["rating_num"].apply(create_sentiment_label)

    df["full_review"] = (df["Review Title"] + " " + df["Review Text"]).str.strip()

    df["clean_text"] = df["full_review"].apply(clean_text)

    return df