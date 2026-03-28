import streamlit as st
import joblib
import re
import string

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Customer Review Analyzer",
    layout="centered"
)

# ---------------------------
# Load model and vectorizer
# ---------------------------
model = joblib.load("outputs/results/model.pkl")
vectorizer = joblib.load("outputs/results/vectorizer.pkl")

# ---------------------------
# Text cleaning function
# ---------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Session state
# ---------------------------
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""

# ---------------------------
# App title
# ---------------------------
st.title("🧠 Customer Review Intelligence System")
st.markdown("### 🚀 Analyse customer sentiment instantly using AI")
st.write("Enter a customer review to analyse sentiment.")

# ---------------------------
# Example and clear buttons
# ---------------------------
example_text = "My order never arrived and customer service was terrible"

col1, col2 = st.columns(2)

with col1:
    if st.button("Try Example"):
        st.session_state["review_text"] = example_text

with col2:
    if st.button("Clear"):
        st.session_state["review_text"] = ""

# ---------------------------
# User input
# ---------------------------
user_input = st.text_area(
    "✍️ Enter Review:",
    value=st.session_state["review_text"],
    height=180
)

# ---------------------------
# Analyse sentiment
# ---------------------------
if st.button("Analyse"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analysing review..."):
            cleaned = clean_text(user_input)
            text_vec = vectorizer.transform([cleaned])

            prediction = model.predict(text_vec)[0]
            probs = model.predict_proba(text_vec)[0]
            confidence = max(probs)

        st.markdown("## 📊 Analysis Result")

        if prediction == "Negative":
            st.error(f"🔴 Sentiment: Negative (Confidence: {confidence:.2f})")
            st.write("💡 Insight: Likely issue related to delivery, refund, or customer service.")

        elif prediction == "Positive":
            st.success(f"🟢 Sentiment: Positive (Confidence: {confidence:.2f})")
            st.write("💡 Insight: Customer is satisfied with the service.")

        else:
            st.info(f"🟡 Sentiment: Neutral (Confidence: {confidence:.2f})")
            st.write("💡 Insight: Mixed or average customer experience.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Built by Noor Saba | AI Customer Review Intelligence System")