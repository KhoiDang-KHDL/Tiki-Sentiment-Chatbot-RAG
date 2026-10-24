"""CSV helpers. Batch outputs require explicit opt-in before overwriting."""
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"


def check_output(path, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite or another output directory.")
    return path


def write_rows(path, fields, rows, overwrite=False):
    path = check_output(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def read_product_ids(path):
    from .client import validate_product_id
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "product_id" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must contain product_id")
        return list(dict.fromkeys(validate_product_id(row["product_id"]) for row in reader if row["product_id"].strip()))
