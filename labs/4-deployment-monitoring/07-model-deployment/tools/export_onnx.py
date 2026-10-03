"""Export DistilBERT (the model text_classifier serves) to ONNX for the ONNX Runtime and TensorRT tasks.

Runs inside the Triton image (it has torch + transformers + onnx): `make export-onnx`.
Writes model_repository/distilbert_onnx/1/model.onnx with dynamic batch and sequence axes.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
out = Path(sys.argv[1] if len(sys.argv) > 1 else "/repo/distilbert_onnx/1/model.onnx")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
enc = tokenizer(["Triton serves this model.", "TensorRT compiles it."], padding="max_length",
                max_length=128, return_tensors="pt")
# INT32 ids: TensorRT and ONNX Runtime both handle them natively, and they halve the input bytes.
ids, mask = enc["input_ids"].int(), enc["attention_mask"].int()

out.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(
    model, (ids, mask), str(out),
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    # Axis 0 = batch (Triton's dynamic batcher varies it), axis 1 = sequence length.
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                  "logits": {0: "batch"}},
    opset_version=17, dynamo=False,
)
with torch.inference_mode():
    print("PyTorch logits:", model(input_ids=ids, attention_mask=mask).logits.tolist())
print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
