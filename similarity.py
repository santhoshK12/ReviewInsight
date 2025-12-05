import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ReviewSimilarity:
    """
    Build embeddings for a set of reviews and
    find top-k most similar reviews to a query.
    """

    def __init__(self, csv_path="data/reviews_imdb_5k.csv", model_name="all-MiniLM-L6-v2"):
        print("Loading reviews for similarity search...")
        df = pd.read_csv(csv_path)

        # keep just text (you could also keep labels if you want later)
        self.texts = df["text"].fillna("").reset_index(drop=True)

        print(f"Loaded {len(self.texts)} reviews.")

        print(f"Loading sentence-transformer model: {model_name} ...")
        self.model = SentenceTransformer(model_name)

        print("Encoding all reviews into embeddings (this may take a bit)...")
        self.embeddings = self.model.encode(
            list(self.texts),
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        print("Embeddings ready.")

    def top_k_similar(self, query: str, k: int = 3):
        """
        Return list of (index, similarity_score, text) for top-k similar reviews.
        """
        if not query or not query.strip():
            return []

        query_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_emb, self.embeddings)[0]

        # indices of top k scores (excluding NaNs)
        top_idx = np.argsort(-sims)[:k]

        results = []
        for i in top_idx:
            results.append((int(i), float(sims[i]), self.texts.iloc[i]))
        return results


if __name__ == "__main__":
    sim = ReviewSimilarity(csv_path="data/reviews_imdb_5k.csv")

    q = "The movie had great acting but the story was boring and too long."
    print("\nQuery:", q)
    matches = sim.top_k_similar(q, k=3)

    for idx, score, text in matches:
        print("\n---")
        print(f"Similarity: {score:.3f}")
        print(f"Review #{idx}: {text[:200]}...")
