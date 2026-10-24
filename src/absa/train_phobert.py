from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
import json
import shutil
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import wandb
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_ROOT = BASE_DIR / "data/absa"
MODEL_DIR = BASE_DIR / "models/absa"

WANDB_PROJECT = "sentiment-chatbot-mlops"

ASPECTS = ["description", "quality", "packaging", "delivery", "service", "price"]
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

MODEL_NAME = "vinai/phobert-base"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(version):
    data_dir = DATA_ROOT / version

    train_df = pd.read_json(data_dir / "train.json")
    dev_df = pd.read_json(data_dir / "dev.json")
    test_df = pd.read_json(data_dir / "test.json")

    return {
        "train_df": train_df,
        "dev_df": dev_df,
        "test_df": test_df,
    }

class ABSADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.encodings = tokenizer(dataframe["text"].tolist(),
                                   padding="max_length",
                                   truncation=True,
                                   max_length=max_length,
                                   return_tensors="pt")
        self.labels = dataframe[ASPECTS].replace(LABEL_MAP).values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class MultiHeadPhoBERT(nn.Module):
    def __init__(self, model_name, hidden_size, num_classes, num_aspects, dropout=0.3):
        super().__init__()

        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, num_classes) for _ in range(num_aspects)])

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)

        logits = []
        for head in self.heads:
            logits.append(head(cls_output))

        return logits

def compute_metrics(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1
    }

def train_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train()

    total_loss = 0

    loop = tqdm(dataloader, desc="Training", leave=False)

    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)

        loss = 0
        for i in range(len(ASPECTS)):
            loss += criterion(outputs[i], labels[:, i])

        loss /= len(ASPECTS)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()

    aspect_predictions = {aspect: [] for aspect in ASPECTS}
    aspect_labels = {aspect: [] for aspect in ASPECTS}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask)

            for i, aspect in enumerate(ASPECTS):
                preds = torch.argmax(outputs[i], dim=1)

                aspect_predictions[aspect].extend(preds.cpu().numpy())
                aspect_labels[aspect].extend(labels[:, i].cpu().numpy())

    all_metrics = {}
    avg_f1 = []

    for aspect in ASPECTS:
        metrics = compute_metrics(aspect_predictions[aspect], aspect_labels[aspect])
        all_metrics[aspect] = metrics
        avg_f1.append(metrics["f1_macro"])

    overall_f1 = sum(avg_f1) / len(avg_f1)

    return all_metrics, overall_f1

def train_model(config):
    wandb.use_artifact(f"absa-dataset:{config.dataset_version}", type="dataset")

    data = load_data(config.dataset_version)
    train_df = data["train_df"]
    dev_df = data["dev_df"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast = False)

    train_dataset = ABSADataset(dataframe=train_df,
                                tokenizer=tokenizer,
                                max_length=config.max_length)
    dev_dataset = ABSADataset(dataframe=dev_df,
                              tokenizer=tokenizer,
                              max_length=config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size)

    model = MultiHeadPhoBERT(model_name=MODEL_NAME,
                             hidden_size=768,
                             num_classes=3,
                             num_aspects=len(ASPECTS),
                             dropout=config.dropout).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        metrics, current_f1 = evaluate(model, dev_loader)

        wandb.log({
            "train_loss": train_loss,
            "avg_f1": current_f1,
            "epoch": epoch + 1
        })

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val F1: {current_f1:.4f}")

        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print("[EARLY STOPPING]")
            break

    model.load_state_dict(best_model_state)
    final_metrics, final_avg_f1 = evaluate(model, dev_loader)
    model_save_dir = MODEL_DIR / config.dataset_version / "phobert"

    current_best = 0

    results_path = model_save_dir / f"phobert_results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            old_results = json.load(f)

        old_scores = [old_results[a]["f1_macro"] for a in ASPECTS]
        current_best = sum(old_scores) / len(old_scores)

    if final_avg_f1 > current_best:
        print(f"[BEST MODEL UPDATED] {current_best:.4f} -> {final_avg_f1:.4f}")

        if model_save_dir.exists():
            shutil.rmtree(model_save_dir)

        model_save_dir.mkdir(parents=True,exist_ok=True)

        checkpoint = {
            "model_state_dict": best_model_state,
            "config": dict(config),
            "metrics": final_metrics,
            "aspects": ASPECTS,
            "model_name": MODEL_NAME
        }

        torch.save(checkpoint, model_save_dir / f"phobert.pt")

        with open(model_save_dir / f"phobert_results.json", "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=4)

        artifact = wandb.Artifact(
            name=f"phobert-{config.dataset_version}",
            type="model",
            metadata=dict(config)
        )
        artifact.add_dir(str(model_save_dir))
        wandb.log_artifact(artifact)

    wandb.finish()

def train_phobert(config=None):
    with wandb.init(config=config, group="phobert", project=WANDB_PROJECT):
        config = wandb.config
        wandb.run.name = (f"phobert-{config.dataset_version}-lr{config.learning_rate}-bs{config.batch_size}-drop{config.dropout}")

        train_model(config)

sweep_config = {
    "method": "grid",

    "metric": {
        "name": "avg_f1",
        "goal": "maximize"
    },

    "parameters": {
        "dataset_version": {"values": ["v2"]},
        "learning_rate": {"values": [2e-5, 5e-5]},
        "batch_size": {"values": [16, 32]},
        "epochs": {"values": [20]},
        "max_length": {"values": [128]},
        "dropout": {"values": [0.3, 0.5]},
        "patience": {"values": [5]}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=train_phobert)
