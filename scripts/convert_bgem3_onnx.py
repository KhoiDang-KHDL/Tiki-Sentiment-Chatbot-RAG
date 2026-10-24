"""
Convert bge-m3 dense encoder to ONNX.
Sparse vectors vẫn dùng FlagEmbedding (không export được sparse head).
Dense ONNX sẽ nhanh hơn 3-4x cho retrieval.
"""
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

ONNX_DIR = Path("/opt/airflow/models/bge-m3-onnx")
MODEL_NAME = "BAAI/bge-m3"


def main():
    print("Loading bge-m3 base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Dummy input
    dummy = tokenizer(["test query"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ONNX_DIR / "bge_m3_dense.onnx"

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    tokenizer.save_pretrained(str(ONNX_DIR))

    # Verify
    print("Verifying...")
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    onnx_out = session.run(None, {
        "input_ids": dummy["input_ids"].numpy().astype(np.int64),
        "attention_mask": dummy["attention_mask"].numpy().astype(np.int64),
    })[0]

    with torch.no_grad():
        pt_out = model(dummy["input_ids"], dummy["attention_mask"]).last_hidden_state.numpy()

    diff = abs(pt_out - onnx_out).max()
    print(f"Max diff: {diff:.6f}")
    print(f"ONNX file: {onnx_path.stat().st_size / 1024 / 1024:.1f}MB")
    print("Done!")


if __name__ == "__main__":
    main()
