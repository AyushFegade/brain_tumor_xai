
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.core import (
    compute_metrics, expected_calibration_error, get_transforms,
    load_bundle, multiclass_brier, selective_metrics, softmax_with_temperature
)

def main():
    p = argparse.ArgumentParser(
        description="Optional external-folder evaluation. The external folder must contain the same four class folders."
    )
    p.add_argument("--bundle", default="outputs/best_model_bundle.pth")
    p.add_argument("--external-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", default="outputs/external_validation.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_bundle(args.bundle, device)
    _, eval_tfms = get_transforms(meta["image_size"])
    ds = datasets.ImageFolder(args.external_dir, transform=eval_tfms)

    if ds.classes != meta["class_names"]:
        raise ValueError(f"Expected classes {meta['class_names']}, got {ds.classes}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    logits_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            logits_all.append(model(x.to(device)).cpu())
            labels_all.append(y)

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()
    probs = softmax_with_temperature(logits, meta["temperature"]).numpy()

    result = {
        **compute_metrics(labels, probs),
        "ece": expected_calibration_error(probs, labels),
        "brier": multiclass_brier(probs, labels),
        "selective": selective_metrics(probs, labels, meta["confidence_threshold"]),
        "n": len(ds),
        "classes": ds.classes,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
