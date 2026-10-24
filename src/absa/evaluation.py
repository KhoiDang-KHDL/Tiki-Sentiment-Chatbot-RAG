from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
import json
from pathlib import Path
import pandas as pd
import wandb
import joblib
import torch
import time
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from src.absa.train_dl import ABSADataset as DL_ABSADataset, MultiHeadBiGRU, MultiHeadTextCNN
from src.absa.train_phobert import ABSADataset as PhoBERT_ABSADataset, MultiHeadPhoBERT
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_ROOT = BASE_DIR / "data/absa"
MODEL_DIR = BASE_DIR / "models/absa"
RESULT_PATH = (MODEL_DIR / "test_results.json")

WANDB_PROJECT = "sentiment-chatbot-mlops"

ASPECTS = ["description", "quality", "packaging", "delivery", "service", "price"]
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def evaluate_ml_model(model_name, dataset_version="v1"):
    start_time = time.time()

    with wandb.init(project=WANDB_PROJECT, group="evaluation", job_type="evaluation", name=f"eval-{model_name}-{dataset_version}") as run:
        dataset_artifact = run.use_artifact(f"absa-dataset:{dataset_version}", type="dataset")
        dataset_dir = Path(dataset_artifact.download())
        test_df = pd.read_json(dataset_dir / "test.json")

        model_artifact = run.use_artifact(f"{model_name}-{dataset_version}:latest", type="model")
        model_dir = Path(model_artifact.download())
        checkpoint = joblib.load(model_dir / f"{model_name}.pkl")
        vectorizer, models = checkpoint["vectorizer"], checkpoint["models"]

        X_test = vectorizer.transform(test_df["text"].tolist())

        all_metrics, avg_f1 = {}, []
        for aspect in ASPECTS:
            model = models[aspect]
            y_true = [LABEL_MAP[label] for label in test_df[aspect]]
            y_pred = model.predict(X_test)

            metrics = compute_metrics(y_pred, y_true)
            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(
                y_true, y_pred,
                target_names=["negative", "neutral", "positive"],
                output_dict=True, zero_division=0
            )

            all_metrics[aspect] = {
                "metrics": metrics,
                "confusion_matrix": cm.tolist(),
                "classification_report": report
            }
            avg_f1.append(metrics["f1_macro"])

            wandb.log({
                f"{aspect}/accuracy": metrics["accuracy"],
                f"{aspect}/precision_macro": metrics["precision_macro"],
                f"{aspect}/recall_macro": metrics["recall_macro"],
                f"{aspect}/f1_macro": metrics["f1_macro"]
            })
            wandb.log({
                f"{aspect}/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=y_true, preds=y_pred,
                    class_names=["negative", "neutral", "positive"]
                )
            })

        overall_f1 = sum(avg_f1) / len(avg_f1)
        eval_time = time.time() - start_time
        final_result = {
            "model_name": model_name,
            "dataset_version": dataset_version,
            "avg_f1_macro": overall_f1,
            "evaluation_time_seconds": round(eval_time, 2),
            "aspect_results": all_metrics
        }
        wandb.summary["avg_f1_macro"] = overall_f1


        print(f"{model_name.upper()} TEST RESULT")

        for aspect in ASPECTS:
            print(f"\n[{aspect}]")

            for metric_name, value in all_metrics[aspect]["metrics"].items():
                print(f"{metric_name}: {value:.4f}")

            print("\nConfusion Matrix:")
            print(all_metrics[aspect]["confusion_matrix"])

        print(f"Avg F1: {overall_f1:.4f}")

        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if RESULT_PATH.exists():
            with open(RESULT_PATH, "r", encoding="utf-8") as f:
                all_saved_results = json.load(f)

        else:
            all_saved_results = {}

        result_key = f"{model_name}_{dataset_version}"
        all_saved_results[result_key] = final_result
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_saved_results, f, ensure_ascii=False, indent=4)

        single_result_path = MODEL_DIR / dataset_version / f"{model_name}" / f"{model_name}_evaluation.json"
        with open(single_result_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)
        eval_artifact = wandb.Artifact(
            name=f"{model_name}-evaluation-{dataset_version}",
            type="evaluation",
            metadata={"avg_f1_macro": overall_f1}
        )
        eval_artifact.add_file(str(single_result_path))
        run.log_artifact(eval_artifact)

        return final_result

def evaluate_dl_model(model_name, dataset_version="v1"):
    start_time = time.time()

    with wandb.init(project=WANDB_PROJECT, group="evaluation", job_type="evaluation", name=f"eval-{model_name}-{dataset_version}") as run:
        dataset_artifact = run.use_artifact(f"absa-dataset:{dataset_version}", type="dataset")
        dataset_dir = Path(dataset_artifact.download())
        test_df = pd.read_json(dataset_dir / "test.json")

        model_artifact = run.use_artifact(f"{model_name}-{dataset_version}:latest", type="model")
        model_dir = Path(model_artifact.download())
        checkpoint = torch.load(model_dir / f"{model_name}.pt", map_location=DEVICE)
        vocab, config = checkpoint["vocab"], checkpoint["config"]
        wandb.config.update(config)

        test_dataset = DL_ABSADataset(texts=test_df["text"].tolist(), dataframe=test_df, vocab=vocab, max_length=config["max_length"])
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

        if model_name == "cnn":
            model = MultiHeadTextCNN(vocab_size=len(vocab), embed_dim=config["embed_dim"], num_classes=3, num_aspects=len(ASPECTS)).to(DEVICE)

        elif model_name == "gru":
            model = MultiHeadBiGRU(vocab_size=len(vocab), embed_dim=config["embed_dim"], hidden_size=config["hidden_size"], num_classes=3, num_aspects=len(ASPECTS)).to(DEVICE)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        aspect_predictions, aspect_labels = {a: [] for a in ASPECTS}, {a: [] for a in ASPECTS}

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids, labels = batch["input_ids"].to(DEVICE), batch["labels"].to(DEVICE)
                outputs = model(input_ids)

                for i, aspect in enumerate(ASPECTS):
                    preds = torch.argmax(outputs[i], dim=1)
                    aspect_predictions[aspect].extend(preds.cpu().numpy())
                    aspect_labels[aspect].extend(labels[:, i].cpu().numpy())

        all_metrics, avg_f1 = {}, []
        for aspect in ASPECTS:
            y_true, y_pred = aspect_labels[aspect], aspect_predictions[aspect]
            metrics, cm = compute_metrics(y_pred, y_true), confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=["negative","neutral","positive"], output_dict=True, zero_division=0)
            all_metrics[aspect] = {"metrics": metrics, "confusion_matrix": cm.tolist(), "classification_report": report}
            avg_f1.append(metrics["f1_macro"])
            wandb.log({
                f"{aspect}/accuracy": metrics["accuracy"],
                f"{aspect}/precision_macro": metrics["precision_macro"],
                f"{aspect}/recall_macro": metrics["recall_macro"],
                f"{aspect}/f1_macro": metrics["f1_macro"],
                f"{aspect}/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=["negative", "neutral", "positive"]
                )
            })

        overall_f1, eval_time = sum(avg_f1)/len(avg_f1), time.time() - start_time
        final_result = {"model_name": model_name,
                        "dataset_version": dataset_version,
                        "avg_f1_macro": overall_f1,
                        "evaluation_time_seconds": round(eval_time,2),
                        "aspect_results": all_metrics}
        wandb.summary["avg_f1_macro"] = overall_f1

        print(f"{model_name.upper()} TEST RESULT")
        for aspect in ASPECTS:
            print(f"\n[{aspect}]")
            for m, v in all_metrics[aspect]["metrics"].items():
                print(f"{m}: {v:.4f}")
            print("\nConfusion Matrix:"); print(all_metrics[aspect]["confusion_matrix"])
        print(f"\nAvg F1: {overall_f1:.4f}")

        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if RESULT_PATH.exists():
            with open(RESULT_PATH, "r", encoding="utf-8") as f:
                all_saved_results = json.load(f)

        else:
            all_saved_results = {}

        all_saved_results[f"{model_name}_{dataset_version}"] = final_result
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_saved_results, f, ensure_ascii=False, indent=4)

        single_result_path = MODEL_DIR / dataset_version / model_name / f"{model_name}_evaluation.json"
        single_result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(single_result_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)

        eval_artifact = wandb.Artifact(name=f"{model_name}-evaluation-{dataset_version}", type="evaluation", metadata={"avg_f1_macro": overall_f1})
        eval_artifact.add_file(str(single_result_path))
        run.log_artifact(eval_artifact)

        return final_result

def evaluate_phobert(dataset_version = "v1"):
    start_time = time.time()

    with wandb.init(project=WANDB_PROJECT, group="evaluation", job_type="evaluation", name=f"eval-phobert-{dataset_version}") as run:
        dataset_artifact = run.use_artifact(f"absa-dataset:{dataset_version}", type="dataset")
        dataset_dir = Path(dataset_artifact.download())
        test_df = pd.read_json(dataset_dir / "test.json")

        model_artifact = run.use_artifact(f"phobert-{dataset_version}:latest", type="model")
        model_dir = Path(model_artifact.download())
        checkpoint = torch.load(model_dir / "phobert.pt", map_location=DEVICE, weights_only=False)
        config = checkpoint["config"]
        wandb.config.update(config)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"], use_fast=False)
        test_dataset = PhoBERT_ABSADataset(dataframe=test_df, tokenizer=tokenizer, max_length=config["max_length"])
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

        model = MultiHeadPhoBERT(model_name=checkpoint["model_name"], hidden_size=768, num_classes=3, num_aspects=len(ASPECTS), dropout=config["dropout"]).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        aspect_predictions, aspect_labels = {a: [] for a in ASPECTS}, {a: [] for a in ASPECTS}

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids, attention_mask, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE)
                outputs = model(input_ids, attention_mask)

                for i, aspect in enumerate(ASPECTS):
                    preds = torch.argmax(outputs[i], dim=1)
                    aspect_predictions[aspect].extend(preds.cpu().numpy())
                    aspect_labels[aspect].extend(labels[:, i].cpu().numpy())

        all_metrics, avg_f1 = {}, []
        for aspect in ASPECTS:
            y_true, y_pred = aspect_labels[aspect], aspect_predictions[aspect]
            metrics, cm = compute_metrics(y_pred, y_true), confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=["negative","neutral","positive"], output_dict=True, zero_division=0)
            all_metrics[aspect] = {"metrics": metrics, "confusion_matrix": cm.tolist(), "classification_report": report}
            avg_f1.append(metrics["f1_macro"])
            wandb.log({
                f"{aspect}/accuracy": metrics["accuracy"],
                f"{aspect}/precision_macro": metrics["precision_macro"],
                f"{aspect}/recall_macro": metrics["recall_macro"],
                f"{aspect}/f1_macro": metrics["f1_macro"],
                f"{aspect}/confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred, class_names=["negative","neutral","positive"])
            })

        overall_f1, eval_time = sum(avg_f1)/len(avg_f1), time.time()-start_time
        final_result = {"model_name": "phobert",
                        "dataset_version": dataset_version,
                        "avg_f1_macro": overall_f1,
                        "evaluation_time_seconds": round(eval_time,2),
                        "aspect_results": all_metrics}
        wandb.summary["avg_f1_macro"] = overall_f1

        print("PHOBERT TEST RESULT")
        for aspect in ASPECTS:
            print(f"\n[{aspect}]")
            for m, v in all_metrics[aspect]["metrics"].items():
                print(f"{m}: {v:.4f}")
            print("\nConfusion Matrix:"); print(all_metrics[aspect]["confusion_matrix"])
        print(f"\nAvg F1: {overall_f1:.4f}")

        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if RESULT_PATH.exists():
            with open(RESULT_PATH, "r", encoding="utf-8") as f:
                all_saved_results = json.load(f)

        else:
            all_saved_results = {}

        all_saved_results[f"phobert_{dataset_version}"] = final_result
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_saved_results, f, ensure_ascii=False, indent=4)

        single_result_path = MODEL_DIR / dataset_version / "phobert" / f"phobert_evaluation.json"
        single_result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(single_result_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)

        eval_artifact = wandb.Artifact(name=f"phobert-evaluation-{dataset_version}", type="evaluation", metadata={"avg_f1_macro": overall_f1})
        eval_artifact.add_file(str(single_result_path))
        run.log_artifact(eval_artifact)

    return final_result
