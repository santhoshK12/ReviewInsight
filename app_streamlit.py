import pandas as pd
import torch
import streamlit as st

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from aspects import extract_aspects
from similarity import ReviewSimilarity


# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_sentiment_model(model_dir="models/distilbert_imdb"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def load_similarity_engine(csv_path="data/reviews_imdb_5k.csv"):
    sim = ReviewSimilarity(csv_path=csv_path)
    return sim


def predict_sentiment(text, tokenizer, model, device):
    encodings = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_label_id = probs.argmax()
    confidence = float(probs[pred_label_id])
    label_str = "Positive" if pred_label_id == 1 else "Negative"

    return label_str, confidence


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="ReviewInsight", page_icon="📝")

    st.title("ReviewInsight – Review Intelligence Demo")
    st.write(
        "Enter a movie/product review below. "
        "The system will predict sentiment, extract key aspects, "
        "and show similar reviews from the IMDB dataset."
    )

    # Load models (lazy load when first needed)
    with st.spinner("Loading models..."):
        tokenizer, model, device = load_sentiment_model()
        sim_engine = load_similarity_engine()

    user_text = st.text_area("Enter review text", height=200)

    if st.button("Analyze"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
            return

        # 1) Sentiment
        label, conf = predict_sentiment(user_text, tokenizer, model, device)
        st.subheader("Sentiment")
        st.write(f"**{label}** (confidence: `{conf:.3f}`)")

        # 2) Aspects
        st.subheader("Key Aspects")
        aspects = extract_aspects(user_text, top_k=5)
        if aspects:
            st.write(", ".join(aspects))
        else:
            st.write("_No strong aspects detected._")

        # 3) Similar reviews
        st.subheader("Similar Reviews from IMDB")
        matches = sim_engine.top_k_similar(user_text, k=3)
        if not matches:
            st.write("_No similar reviews found._")
        else:
            for idx, score, text in matches:
                st.markdown(f"**Similarity: `{score:.3f}`**")
                st.write(text)
                st.markdown("---")


if __name__ == "__main__":
    main()
