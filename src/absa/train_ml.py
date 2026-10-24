from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import wandb
import json
import shutil
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_ROOT = BASE_DIR / "data/absa"
MODEL_DIR = BASE_DIR / "models/absa"

WANDB_PROJECT = "sentiment-chatbot-mlops"

ASPECTS = ["description", "quality", "packaging", "delivery", "service", "price"]
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

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

def prepare_data(train_df, dev_df, test_df, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(train_df["text"])
    X_dev = vectorizer.transform(dev_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = {aspect: train_df[aspect].map(LABEL_MAP) for aspect in ASPECTS}
    y_dev = {aspect: dev_df[aspect].map(LABEL_MAP) for aspect in ASPECTS}
    y_test = {aspect: test_df[aspect].map(LABEL_MAP) for aspect in ASPECTS}

    return {
        "vectorizer": vectorizer,
        "X_train": X_train,
        "X_dev": X_dev,
        "X_test": X_test,
        "y_train": y_train,
        "y_dev": y_dev,
        "y_test": y_test
    }

def evaluate_model(predictions, labels):
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

def train_model(model_name, model_builder, config):
    wandb.use_artifact(f"absa-dataset:{config.dataset_version}", type="dataset")

    data = load_data(config.dataset_version)
    train_df = data['train_df']
    dev_df = data['dev_df']
    test_df = data['test_df']

    prepared = prepare_data(train_df, dev_df, test_df, config.max_features)
    vectorizer = prepared['vectorizer']
    X_train = prepared['X_train']
    X_dev = prepared['X_dev']
    y_train = prepared['y_train']
    y_dev = prepared['y_dev']

    model_save_dir = MODEL_DIR / config.dataset_version / model_name
    model_save_dir.mkdir(parents = True, exist_ok = True)

    all_results = {}
    avg_f1_scores = []
    trained_models = {}

    for aspect in ASPECTS:
        model = model_builder(config)

        model.fit(X_train, y_train[aspect])

        predictions = model.predict(X_dev)
        metrics = evaluate_model(predictions, y_dev[aspect])

        trained_models[aspect] = model
        avg_f1_scores.append(metrics["f1_macro"])
        all_results[aspect] = metrics

        wandb.log({
            f"{aspect}_accuracy": metrics["accuracy"],
            f"{aspect}_precision": metrics["precision_macro"],
            f"{aspect}_recall": metrics["recall_macro"],
            f"{aspect}_f1_macro": metrics["f1_macro"]
        })

    avg_f1 = sum(avg_f1_scores) / len(avg_f1_scores)
    wandb.log({"avg_f1": avg_f1})
    wandb.run.summary["avg_f1"] = avg_f1

    current_best = 0

    results_path = model_save_dir / f"{model_name}_results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            old_results = json.load(f)

        old_f1_scores = [old_results[a]["f1_macro"] for a in ASPECTS]
        current_best = sum(old_f1_scores) / len(old_f1_scores)

    if avg_f1 > current_best:
        print(f"[BEST MODEL UPDATED] "
              f"{current_best:.4f} -> {avg_f1:.4f}")

        if model_save_dir.exists():
            shutil.rmtree(model_save_dir)

        model_save_dir.mkdir(parents=True, exist_ok=True)

        model_bundle = {
            "vectorizer": vectorizer,
            "models": trained_models,
            "aspects": ASPECTS,
            "label_map": LABEL_MAP,
            "config": dict(config),
            "metrics": all_results
        }

        joblib.dump(model_bundle, model_save_dir / f"{model_name}.pkl")
        with open(model_save_dir / f"{model_name}_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        artifact = wandb.Artifact(
            name=f"{model_name}-{config.dataset_version}",
            type="model",
            metadata=dict(config))
        artifact.add_dir(str(model_save_dir))
        wandb.log_artifact(artifact)

    wandb.finish()

def train_logreg(config=None):
    with wandb.init(config = config, group = "logreg", project = WANDB_PROJECT):
        config = wandb.config
        run_name = (f"logreg-"f"{config.dataset_version}-"f"mf{config.max_features}-"f"C{config.C}")
        wandb.run.name = run_name

        train_model(model_name="logreg",
                    model_builder=lambda cfg: LogisticRegression(C=cfg.C, max_iter=1000, random_state=42, class_weight="balanced"),
                    config=config
        )

def train_rf(config=None):
    with wandb.init(config=config, group = "rf", project = WANDB_PROJECT):
        config = wandb.config
        run_name = (f"rf-"f"{config.dataset_version}-"f"mf{config.max_features}-"f"est{config.n_estimators}-"f"depth{config.max_depth}")
        wandb.run.name = run_name

        train_model(model_name="random_forest",
                    model_builder=lambda cfg: RandomForestClassifier(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, min_samples_split=cfg.min_samples_split, random_state=42, class_weight="balanced"),
                    config=config
        )

logreg_sweep_config = {
    "method": "grid",

    "metric": {
        "name": "avg_f1",
        "goal": "maximize"
    },

    "parameters": {
        "dataset_version": {
            "values": ["v2"]
        },
        "max_features": {
            "values": [5000, 10000]
        },
        "C": {
            "values": [0.1, 1, 10]
        }
    }
}

rf_sweep_config = {
    "method": "grid",

    "metric": {
        "name": "avg_f1",
        "goal": "maximize"
    },

    "parameters": {
        "dataset_version": {
            "values": ["v2"]
        },
        "max_features": {
            "values": [5000, 10000]
        },
        "n_estimators": {
            "values": [100, 200]
        },
        "max_depth": {
            "values": [None, 10, 20]
        },
        "min_samples_split": {
            "values": [2, 5]
        }
    }
}

if __name__ == "__main__":
    logreg_sweep_id = wandb.sweep(logreg_sweep_config, project=WANDB_PROJECT)
    wandb.agent(logreg_sweep_id, function=train_logreg)

    rf_sweep_id = wandb.sweep(rf_sweep_config, project=WANDB_PROJECT)
    wandb.agent(rf_sweep_id, function=train_rf)
