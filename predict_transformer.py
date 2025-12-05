import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


def load_model(model_dir="models/distilbert_imdb"):
    """
    Load the fine-tuned DistilBERT model and tokenizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_sentiment(text: str, tokenizer, model, device):
    """
    Predict sentiment for a single text.
    Returns: (label_str, confidence_float)
    """
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

    # IMDB labels: 0 = negative, 1 = positive
    label_str = "Positive" if pred_label_id == 1 else "Negative"

    return label_str, confidence


if __name__ == "__main__":
    tokenizer, model, device = load_model()

    while True:
        text = input("\nEnter a review (or 'q' to quit): ").strip()
        if text.lower() == "q":
            print("Exiting.")
            break
        if not text:
            print("Please enter some text.")
            continue

        label, conf = predict_sentiment(text, tokenizer, model, device)
        print(f"Prediction: {label} (confidence: {conf:.4f})")
