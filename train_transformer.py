import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from dataset import load_review_data


# -----------------------------
# Dataset wrapper
# -----------------------------
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# -----------------------------
# Training & evaluation helpers
# -----------------------------
def train_epoch(model, data_loader, optimizer, scheduler, device, epoch_idx=0):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(data_loader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # progress log
        if step % 50 == 0 or step == 1:
            avg_so_far = total_loss / step
            print(f"  [Epoch {epoch_idx+1}] Step {step}/{len(data_loader)} - "
                  f"loss: {avg_so_far:.4f}")

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def eval_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    return acc, precision, recall, f1


# -----------------------------
# Main training function
# -----------------------------
def train_distilbert(
    csv_path="data/reviews_imdb_5k.csv",
    model_save_dir="models/distilbert_imdb",
    batch_size=8,
    num_epochs=1,        # 1 epoch for now
    max_len=128,        # shorter sequences -> faster
    lr=2e-5,
    subset_size=2000,   # use 2000 samples for speed
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )
    model.to(device)

    # Load data
    df = load_review_data(csv_path)
    print(f"Original dataset size: {len(df)}")

    # take a smaller subset for faster training
    if subset_size is not None and len(df) > subset_size:
        df = df.sample(n=subset_size, random_state=42).reset_index(drop=True)
        print(f"Using subset of size: {len(df)}")

    texts = df["text"]
    labels = df["label"]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"Train size: {len(train_texts)}")
    print(f"Val size:   {len(val_texts)}")

    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    best_metrics = None

    for epoch in range(num_epochs):
        print(f"\n========== Epoch {epoch + 1}/{num_epochs} ==========")
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch_idx=epoch
        )
        print(f"Train loss: {train_loss:.4f}")

        acc, precision, recall, f1 = eval_model(model, val_loader, device)
        print("Validation metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = (acc, precision, recall, f1)

    os.makedirs(model_save_dir, exist_ok=True)
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"\nModel & tokenizer saved to: {model_save_dir}")

    if best_metrics is not None:
        acc, precision, recall, f1 = best_metrics
        print("\nBest validation metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

    return best_metrics


if __name__ == "__main__":
    train_distilbert()
