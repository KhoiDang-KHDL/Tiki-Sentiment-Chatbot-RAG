"""
Convert PhoBERT ABSA model to ONNX format.
Run: docker exec tiki-airflow-worker python /opt/airflow/convert_phobert_onnx.py
"""
import os
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# Paths trong airflow-worker container
MODEL_DIR = Path("/opt/airflow/models/absa/v2/phobert")
ONNX_DIR = Path("/opt/airflow/models/absa/v2/phobert_onnx")
MODEL_NAME = "vinai/phobert-base"
ASPECTS = ["description", "quality", "packaging", "delivery", "service", "price"]


class MultiHeadPhoBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleList([
            nn.Linear(768, 3) for _ in range(len(ASPECTS))
        ])

    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = self.dropout(out.last_hidden_state[:, 0])
        logits = torch.stack([head(cls_out) for head in self.heads], dim=1)
        return logits


def main():
    print("Loading PyTorch model...")
    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_DIR / "phobert.pt", map_location=device)

    model = MultiHeadPhoBERT()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    # Dummy input
    dummy_text = ["sản phẩm chất lượng tốt"]
    enc = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Export ONNX
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ONNX_DIR / "phobert_absa.onnx"

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Save tokenizer
    tokenizer.save_pretrained(str(ONNX_DIR))

    # Verify
    print("Verifying ONNX model...")
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    outputs = session.run(None, {
        "input_ids": enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy(),
    })
    logits_onnx = outputs[0]
    print(f"ONNX output shape: {logits_onnx.shape}")

    # Compare
    with torch.no_grad():
        logits_pt = model(enc["input_ids"], enc["attention_mask"]).numpy()

    diff = abs(logits_pt - logits_onnx).max()
    print(f"Max diff PyTorch vs ONNX: {diff:.6f}")
    if diff < 0.01:
        print("PASS — ONNX model matches PyTorch")
    else:
        print("WARNING — outputs differ significantly")

    # File size
    pt_size = (MODEL_DIR / "phobert.pt").stat().st_size / 1024 / 1024
    onnx_size = onnx_path.stat().st_size / 1024 / 1024
    print(f"\nPyTorch: {pt_size:.1f}MB -> ONNX: {onnx_size:.1f}MB")
    print("Done!")


if __name__ == "__main__":
    main()
