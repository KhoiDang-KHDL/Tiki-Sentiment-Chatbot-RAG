from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
import json
import shutil
from collections import Counter
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from underthesea import word_tokenize
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_ROOT = BASE_DIR / "data/absa"
MODEL_DIR = BASE_DIR / "models/absa"

WANDB_PROJECT = "sentiment-chatbot-mlops"

ASPECTS = ["description", "quality", "packaging", "delivery", "service", "price"]
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

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

def tokenize(text):
    return word_tokenize(text, format="text").split()

def build_vocab(texts, min_freq=2):
    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab

def encode_text(text, vocab, max_length):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    ids = ids[:max_length]

    if len(ids) < max_length:
        ids += [vocab[PAD_TOKEN]] * (max_length - len(ids))

    return ids

class ABSADataset(Dataset):
    def __init__(self, texts, dataframe, vocab, max_length):
        self.texts = texts
        self.dataframe = dataframe
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = encode_text(self.texts[idx], self.vocab, self.max_length)
        labels = [LABEL_MAP[self.dataframe.iloc[idx][aspect]] for aspect in ASPECTS]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
            }

class MultiHeadTextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_aspects):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.shared_dim = 256
        self.heads = nn.ModuleList([nn.Linear(self.shared_dim, num_classes) for _ in range(num_aspects)])

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x1 = torch.max(x1, dim=2)[0]
        x2 = torch.max(x2, dim=2)[0]
        x = torch.cat([x1, x2], dim=1)
        x = self.dropout(x)

        outputs = []

        for head in self.heads:
            outputs.append(head(x))

        return outputs

class MultiHeadBiGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_aspects):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.shared_dim = hidden_size * 2
        self.heads = nn.ModuleList([nn.Linear(self.shared_dim, num_classes) for _ in range(num_aspects)])

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, hidden = self.gru(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)

        outputs = []

        for head in self.heads:
            outputs.append(head(hidden))

        return outputs

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

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()

    total_loss = 0

    loop = tqdm(dataloader, desc="Training", leave=False)

    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids)

        loss = 0

        for i in range(len(ASPECTS)):
            loss += criterion(outputs[i], labels[:, i])

        loss /= len(ASPECTS)
        loss.backward()

        optimizer.step()

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
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids)

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

def train_model(model_name, model_builder, config):
    wandb.use_artifact(f"absa-dataset:{config.dataset_version}", type="dataset")

    data = load_data(config.dataset_version)
    train_df = data["train_df"]
    dev_df = data["dev_df"]

    vocab = build_vocab(train_df["text"].tolist())

    train_dataset = ABSADataset(texts=train_df["text"].tolist(),
                                dataframe=train_df,
                                vocab=vocab,
                                max_length=config.max_length)
    dev_dataset = ABSADataset(texts=dev_df["text"].tolist(),
                              dataframe=dev_df,
                              vocab=vocab,
                              max_length=config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size)

    model = model_builder(len(vocab), config).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion)
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
    model_save_dir = MODEL_DIR / config.dataset_version / model_name

    current_best = 0

    results_path = model_save_dir / f"{model_name}_results.json"
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
            "vocab": vocab,
            "model_state_dict": best_model_state,
            "config": dict(config),
            "metrics": final_metrics,
            "aspects": ASPECTS
        }

        torch.save(checkpoint, model_save_dir / f"{model_name}.pt")

        with open(model_save_dir / f"{model_name}_results.json", "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=4)

        artifact = wandb.Artifact(
            name=f"{model_name}-{config.dataset_version}",
            type="model",
            metadata=dict(config)
        )
        artifact.add_dir(str(model_save_dir))
        wandb.log_artifact(artifact)

    wandb.finish()

def train_cnn(config=None):
    with wandb.init(config=config, group="cnn",project=WANDB_PROJECT):
        config = wandb.config
        wandb.run.name = (
            f"cnn-"
            f"{config.dataset_version}-"
            f"lr{config.learning_rate}-"
            f"emb{config.embed_dim}"
        )

        train_model(model_name="cnn",
                    model_builder=lambda vocab_size, cfg: MultiHeadTextCNN(vocab_size=vocab_size,
                                                                           embed_dim=cfg.embed_dim,
                                                                           num_classes=3,
                                                                           num_aspects=len(ASPECTS)),
                    config=config)

def train_gru(config=None):
    with wandb.init(config=config, group="gru", project=WANDB_PROJECT):
        config = wandb.config
        wandb.run.name = (
            f"gru-"
            f"{config.dataset_version}-"
            f"lr{config.learning_rate}-"
            f"emb{config.embed_dim}-"
            f"hidden{config.hidden_size}"
        )

        train_model(model_name="gru",
                    model_builder=lambda vocab_size, cfg: MultiHeadBiGRU(vocab_size=vocab_size,
                                                                         embed_dim=cfg.embed_dim,
                                                                         hidden_size=cfg.hidden_size,
                                                                         num_classes=3,
                                                                         num_aspects=len(ASPECTS)),
                    config=config)

cnn_sweep_config = {
    "method": "grid",

    "metric": {
        "name": "avg_f1",
        "goal": "maximize"
    },

    "parameters": {
        "dataset_version": {"values": ["v2"]},
        "patience": {"values": [5]},
        "learning_rate": {"values": [1e-3, 5e-4]},
        "batch_size": {"values": [32]},
        "epochs": {"values": [20]},
        "embed_dim": {"values": [128, 256]},
        "max_length": {"values": [128]}
    }
}

gru_sweep_config = {
    "method": "grid",

    "metric": {
        "name": "avg_f1",
        "goal": "maximize"
    },

    "parameters": {
        "dataset_version": {"values": ["v2"]},
        "patience": {"values": [5]},
        "learning_rate": {"values": [1e-3, 5e-4]},
        "batch_size": {"values": [32]},
        "epochs": {"values": [20]},
        "embed_dim": {"values": [128, 256]},
        "hidden_size": {"values": [128, 256]},
        "max_length": {"values": [128]}
    }
}

if __name__ == "__main__":
    cnn_sweep_id = wandb.sweep(cnn_sweep_config, project=WANDB_PROJECT)
    wandb.agent(cnn_sweep_id, function=train_cnn)

    gru_sweep_id = wandb.sweep(gru_sweep_config, project=WANDB_PROJECT)
    wandb.agent(gru_sweep_id, function=train_gru)
