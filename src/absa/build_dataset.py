from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
import json
import pandas as pd
from pathlib import Path
import wandb
from sklearn.model_selection import train_test_split

WANDB_PROJECT = "sentiment-chatbot-mlops"

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = BASE_DIR / "data/processed/labeled_processed_products_reviews.csv"
OUTPUT_DIR = BASE_DIR / "data/absa"

def convert_labels(label_str):
    labels = json.loads(label_str)

    row = {
        "description": "none",
        "quality": "none",
        "packaging": "none",
        "delivery": "none",
        "service": "none",
        "price": "none"
    }

    for item in labels:
        aspect = item["aspect"]
        sentiment = item["sentiment"]
        row[aspect] = sentiment

    return row

def build_dataset(df):
    records = []

    for _, row in df.iterrows():
        label_data = convert_labels(row["label"])

        sample = {
            "text": row["clean_content"],
            **label_data
        }

        records.append(sample)

    return pd.DataFrame(records)

def split_dataset(dataset):
    train_df, temp_df = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42
    )

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42
    )

    return train_df, dev_df, test_df

def save_dataset(train_df, dev_df, test_df, full_df, version):
    version_dir = OUTPUT_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_json(
        version_dir / "train.json",
        orient="records",
        force_ascii=False,
        indent=4
    )

    dev_df.to_json(
        version_dir / "dev.json",
        orient="records",
        force_ascii=False,
        indent=4
    )

    test_df.to_json(
        version_dir / "test.json",
        orient="records",
        force_ascii=False,
        indent=4
    )

    full_df.to_json(
        version_dir / "full_dataset.json",
        orient="records",
        force_ascii=False,
        indent=4
    )

    return version_dir

def log_dataset_artifact(version_dir, version):
    wandb.init(
        project=WANDB_PROJECT,
        entity="cs317-mlops",
        job_type="dataset",
        name=f"dataset-build-{version}",
        reinit=True
    )


    artifact = wandb.Artifact(
        name=f"absa-dataset",
        type="dataset",
        metadata={
            "num_samples": len(pd.read_json(version_dir / "full_dataset.json")),
            "num_train": len(pd.read_json(version_dir / "train.json")),
            "num_dev": len(pd.read_json(version_dir / "dev.json")),
            "num_test": len(pd.read_json(version_dir / "test.json"))
        }
    )

    artifact.add_file(str(version_dir / "train.json"))
    artifact.add_file(str(version_dir / "dev.json"))
    artifact.add_file(str(version_dir / "test.json"))
    artifact.add_file(str(version_dir / "full_dataset.json"))

    wandb.log_artifact(artifact)

    wandb.finish()

def update_dataset(old_version, new_version, new_data_path):
    old_dataset_path = (
        OUTPUT_DIR /
        old_version /
        "full_dataset.json"
    )

    old_df = pd.read_json(old_dataset_path)

    new_reviews_df = pd.read_csv(new_data_path)

    new_dataset = build_dataset(new_reviews_df)

    full_dataset = pd.concat(
        [old_df, new_dataset],
        ignore_index=True
    )

    train_df, dev_df, test_df = split_dataset(
        full_dataset
    )

    version_dir = save_dataset(
        train_df,
        dev_df,
        test_df,
        full_dataset,
        new_version
    )

    log_dataset_artifact(
        version_dir,
        new_version
    )

    print(
        f"Updated dataset: {old_version} -> {new_version}"
    )

def main(version):
    df = pd.read_csv(INPUT_PATH)

    dataset = build_dataset(df)

    train_df, dev_df, test_df = split_dataset(dataset)

    version_dir = save_dataset(
        train_df,
        dev_df,
        test_df,
        dataset,
        version
    )

    log_dataset_artifact(
        version_dir,
        version
    )
