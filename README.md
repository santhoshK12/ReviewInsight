# ReviewInsight: Transformer-Based Review Intelligence System

## 1. Project Overview

ReviewInsight is an end-to-end NLP system that analyzes user reviews using modern Transformer models and lightweight linguistic techniques.  
Given any movie or product-style review, the system:

- predicts the sentiment (positive or negative),
- extracts key aspects (topics/entities) mentioned in the text, and
- retrieves semantically similar reviews from a reference dataset.

The project combines a baseline classical ML model with a fine-tuned DistilBERT model, a MiniLM-based semantic similarity component, and a Streamlit web interface for live demonstrations.

---

## 2. Project Goals

1. Build a complete, reproducible NLP pipeline rather than a single model.
2. Compare a traditional TF–IDF + Logistic Regression baseline with a modern Transformer-based model (DistilBERT).
3. Add interpretability through aspect extraction and similar-review retrieval.
4. Deploy the system as an interactive Streamlit app suitable for classroom demonstration.

---

## 3. Dataset

- **Source:** IMDB movie reviews dataset.
- **Labels:** Binary sentiment – positive and negative.
- **Usage in project:**
  - A subset is used to train and evaluate the baseline and DistilBERT sentiment models.
  - A separate 5k-review subset is embedded using MiniLM and used as the corpus for semantic similarity search.
- Minimal preprocessing is applied (lowercasing, basic cleaning) to preserve natural writing style.

---

## 4. System Architecture (High-Level)

The pipeline is organized into four main components:

1. **Sentiment Classification (DistilBERT):** Fine-tuned Transformer model predicts positive/negative sentiment.
2. **Aspect Extraction (spaCy):** Noun and proper-noun extraction provides key aspects of the review.
3. **Semantic Similarity (MiniLM):** Sentence-transformer embeddings plus cosine similarity retrieve the top-k similar reviews.
4. **Streamlit Interface:** Simple web UI that connects all components and displays results in real time.

These components are designed as modular Python scripts so they can be reused or extended.

---

## 5. Implementation Steps (What We Actually Did)

This section explains, step by step, how the project was implemented.

### Step 1 – Environment Setup

- Created a dedicated project folder (`reviewinsight`) with subdirectories:
  - `data/` – datasets (IMDB subsets),
  - `src/` – Python source code,
  - `models/` – saved model weights and embeddings.
- Created a virtual environment and installed required libraries:
  - `transformers`, `torch`, `scikit-learn`, `pandas`, `numpy`,
  - `spacy`, `sentence-transformers`,
  - `streamlit` for the web interface.

### Step 2 – Loading and Inspecting the Dataset

- Implemented a dataset loader (`src/dataset.py`) that:
  - reads the IMDB CSV files,
  - checks for missing values,
  - shuffles and splits the data into train/validation/test sets.
- Verified label distribution to ensure a balanced positive/negative split.

### Step 3 – Baseline Model: TF–IDF + Logistic Regression

- Built a baseline sentiment classifier (`src/train_baseline.py`) using:
  - `TfidfVectorizer` to convert text into sparse feature vectors,
  - `LogisticRegression` as the classifier.
- Trained on the training split and evaluated on the test split.
- Reported metrics (accuracy, precision, recall, F1) as a reference point.
- Saved the trained baseline model using `joblib` for potential reuse.

### Step 4 – DistilBERT Sentiment Model

- Implemented a Transformer-based classifier (`src/train_transformer.py`) using:
  - `DistilBertTokenizer` for tokenization,
  - `DistilBertForSequenceClassification` for fine-tuning.
- Steps performed:
  - Tokenized all reviews with truncation/padding to a fixed length.
  - Created PyTorch datasets and dataloaders for efficient batching.
  - Fine-tuned DistilBERT on the IMDB training set using AdamW optimizer and a learning-rate scheduler.
- Monitored validation loss and accuracy across epochs.
- Saved the fine-tuned model and tokenizer to the `models/` directory.

### Step 5 – Aspect Extraction with spaCy

- Integrated spaCy’s English language model for part-of-speech tagging (`src/aspects.py`).
- For a given review:
  - Performed tokenization and POS tagging.
  - Selected nouns and proper nouns as candidate aspects.
  - Applied lemmatization to normalize words (e.g., “actors” → “actor”).
- Returned a small set of key aspect words for display in the UI.

### Step 6 – Semantic Similarity with MiniLM

- Used a SentenceTransformer MiniLM model to generate dense vector embeddings for reviews (`src/embeddings.py`).
- Workflow:
  - Encoded the 5k-review subset from IMDB into 384-dimensional embeddings.
  - Stored embeddings and corresponding texts in the `models/` directory.
  - For a new user review, generated its embedding and computed cosine similarity against all stored vectors.
  - Selected the top-k most similar reviews to display as “Similar Reviews”.

### Step 7 – Integration of Components

- Implemented a core inference function (`src/pipeline.py`) that:
  - takes a raw review as input,
  - calls the DistilBERT classifier for sentiment,
  - calls the aspect extraction module,
  - calls the similarity engine to retrieve similar reviews,
  - returns a structured result object with:
    - sentiment label and confidence,
    - extracted aspects,
    - list of similar review texts with similarity scores.

### Step 8 – Streamlit Web Application

- Created `app.py` using Streamlit to provide an interactive demo:
  - Single text area where the user pastes a review.
  - “Analyze” button triggers the pipeline.
  - Displays:
    - predicted sentiment and confidence,
    - list of aspects,
    - top-k similar reviews and their similarity scores.
- Used caching where possible to avoid reloading models repeatedly and to keep response time reasonable on CPU.

### Step 9 – Experiments and Validation

- Ran the baseline and DistilBERT models on the same test set.
- Compared classification metrics and observed clear improvements from DistilBERT, especially on nuanced or mixed reviews.
- Verified that the semantic similarity module returns logically related reviews from the IMDB subset.
- Tested edge cases such as short reviews and mixed-polarity reviews.

### Step 10 – Final Demo Preparation

- Prepared the Streamlit app for live presentation.
- Selected example reviews:
  - one clearly positive and one clearly negative with movie names,
  - additional mixed reviews to show model behavior.
- Verified end-to-end functionality: from typing a review to seeing sentiment, aspects, and similar reviews displayed.

---

## 6. How to Run the Project (For Reproducibility)

1. **Clone the repository (once pushed):**
   ```bash
   git clone https://github.com/santhoshK12/reviewinsightNLP.git
   cd reviewinsightNLP
Create environment and install dependencies:

bash
Copy code
pip install -r requirements.txt
(Optional) Re-train models:

Baseline:

bash
Copy code
python src/train_baseline.py
DistilBERT:

bash
Copy code
python src/train_transformer.py
Run the Streamlit app:

bash
Copy code
streamlit run app.py# ReviewInsight
