import spacy

# load small English model once
# this will take a moment the first time
nlp = spacy.load("en_core_web_sm")


def extract_aspects(text: str, top_k: int = 5):
    """
    Very simple aspect extractor:
    - runs spaCy
    - keeps only NOUN / PROPN tokens
    - lemmatizes and lowercases
    - returns top_k most frequent terms
    """
    if not text or not text.strip():
        return []

    doc = nlp(text)

    nouns = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and token.is_alpha
    ]

    if not nouns:
        return []

    freq = {}
    for n in nouns:
        freq[n] = freq.get(n, 0) + 1

    # sort by frequency (high → low)
    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    return [term for term, count in sorted_terms[:top_k]]


if __name__ == "__main__":
    sample = "The phone camera is great, but the battery and speaker quality are terrible."
    aspects = extract_aspects(sample)
    print("Text:", sample)
    print("Aspects:", aspects)
