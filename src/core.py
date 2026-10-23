from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, models, transforms

# LIME imports
from lime import lime_image
from skimage.segmentation import mark_boundaries


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class ProjectConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    head_epochs: int = 3
    fine_tune_epochs: int = 5
    head_lr: float = 1e-3
    fine_tune_lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    target_coverage: float = 0.90


def find_dataset_dirs(dataset_root: str | Path) -> Tuple[Path, Path]:
    root = Path(dataset_root)
    candidates = [
        (root / "Training", root / "Testing"),
        (root / "training", root / "testing"),
    ]
    for train_dir, test_dir in candidates:
        if train_dir.exists() and test_dir.exists():
            return train_dir, test_dir

    for train_dir in root.rglob("Training"):
        test_dir = train_dir.parent / "Testing"
        if test_dir.exists():
            return train_dir, test_dir
    raise FileNotFoundError(
        f"Could not find Training/Testing folders below {root}. "
        "Expected the Kaggle brain-tumor-mri-dataset layout."
    )


def get_transforms(image_size: int = 224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return train_tfms, eval_tfms


def build_loaders(dataset_root: str | Path, cfg: ProjectConfig):
    """Create stratified train/validation/test loaders from all available images."""
    train_dir, test_dir = find_dataset_dirs(dataset_root)
    train_tfms, eval_tfms = get_transforms(cfg.image_size)

    if not np.isclose(cfg.train_size + cfg.val_size + cfg.test_size, 1.0):
        raise ValueError(
            "train_size + val_size + test_size must equal 1.0; got "
            f"{cfg.train_size + cfg.val_size + cfg.test_size:.4f}"
        )
    if min(cfg.train_size, cfg.val_size, cfg.test_size) <= 0:
        raise ValueError("Train, validation, and test ratios must all be greater than zero.")

    base_train = datasets.ImageFolder(train_dir)
    base_test = datasets.ImageFolder(test_dir)
    if base_train.classes != base_test.classes:
        raise ValueError(
            f"Class mismatch: train={base_train.classes}, test={base_test.classes}"
        )

    labels = np.asarray(base_train.targets + base_test.targets, dtype=np.int64)
    idx = np.arange(len(labels))
    train_idx, remaining_idx = train_test_split(
        idx,
        test_size=(cfg.val_size + cfg.test_size),
        random_state=cfg.seed,
        stratify=labels,
    )
    test_fraction_of_remaining = cfg.test_size / (cfg.val_size + cfg.test_size)
    val_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=test_fraction_of_remaining,
        random_state=cfg.seed + 1,
        stratify=labels[remaining_idx],
    )

    train_all = ConcatDataset([
        datasets.ImageFolder(train_dir, transform=train_tfms),
        datasets.ImageFolder(test_dir, transform=train_tfms),
    ])
    eval_all = ConcatDataset([
        datasets.ImageFolder(train_dir, transform=eval_tfms),
        datasets.ImageFolder(test_dir, transform=eval_tfms),
    ])

    train_subset = Subset(train_all, train_idx.tolist())
    val_subset = Subset(eval_all, val_idx.tolist())
    test_subset = Subset(eval_all, test_idx.tolist())

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_subset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin
    )

    class_names = base_train.classes
    split_indices = {"train": train_idx, "validation": val_idx, "test": test_idx}
    split_info = {
        "total_images": int(len(labels)),
        "seed": int(cfg.seed),
        "ratios": {
            "train": float(cfg.train_size),
            "validation": float(cfg.val_size),
            "test": float(cfg.test_size),
        },
        "counts": {name: int(len(values)) for name, values in split_indices.items()},
        "per_class_counts": {
            class_name: {
                split_name: int(np.sum(labels[indices] == class_id))
                for split_name, indices in split_indices.items()
            }
            for class_id, class_name in enumerate(class_names)
        },
        "source_folders_pooled": [str(train_dir), str(test_dir)],
        "original_official_test_partition_preserved": False,
        "split_method": "two-stage stratified train/validation/test split",
    }
    print("Split summary:")
    print(json.dumps({k: split_info[k] for k in ("total_images", "ratios", "counts", "per_class_counts")}, indent=2))
    return train_loader, val_loader, test_loader, class_names, split_info


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "custom_cnn":
        return CustomCNN(num_classes)

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(in_f, num_classes))
        return m

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(in_f, num_classes))
        return m

    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        m = models.densenet121(weights=weights)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, num_classes)
        return m

    raise ValueError(f"Unknown architecture: {name}")


def freeze_backbone(model: nn.Module, name: str):
    name = name.lower()
    if name == "custom_cnn":
        return
    for p in model.parameters():
        p.requires_grad = False

    if name == "resnet50":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif name == "efficientnet_b0":
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif name == "densenet121":
        for p in model.classifier.parameters():
            p.requires_grad = True


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits_list, y_list = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        logits_list.append(logits.detach().cpu())
        y_list.append(y.cpu())
    return torch.cat(logits_list), torch.cat(y_list)


def compute_metrics(y_true, probs):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    preds = probs.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision_macro": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
    }
    try:
        metrics["auc_ovr_macro"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["auc_ovr_macro"] = float("nan")
    return metrics


def train_phase(
    model, train_loader, val_loader, device, epochs, lr,
    weight_decay=1e-4, patience=3
):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        val_logits, val_y = collect_logits(model, val_loader, device)
        val_probs = torch.softmax(val_logits, dim=1).numpy()
        vm = compute_metrics(val_y.numpy(), val_probs)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            **{f"val_{k}": v for k, v in vm.items()}
        }
        history.append(row)

        if vm["f1_macro"] > best_f1:
            best_f1 = vm["f1_macro"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, history, best_f1, best_epoch


def fit_model(model_name, num_classes, loaders, cfg: ProjectConfig, device):
    train_loader, val_loader, _ = loaders
    model = build_model(model_name, num_classes, pretrained=True).to(device)

    histories = []
    if model_name != "custom_cnn":
        freeze_backbone(model, model_name)
        model, h1, _, _ = train_phase(
            model, train_loader, val_loader, device,
            cfg.head_epochs, cfg.head_lr, cfg.weight_decay, cfg.patience
        )
        histories.extend([{**x, "phase": "head"} for x in h1])
        unfreeze_all(model)
        model, h2, best_f1, best_epoch = train_phase(
            model, train_loader, val_loader, device,
            cfg.fine_tune_epochs, cfg.fine_tune_lr, cfg.weight_decay, cfg.patience
        )
        histories.extend([{**x, "phase": "fine_tune"} for x in h2])
    else:
        model, h, best_f1, best_epoch = train_phase(
            model, train_loader, val_loader, device,
            cfg.head_epochs + cfg.fine_tune_epochs, cfg.head_lr,
            cfg.weight_decay, cfg.patience
        )
        histories.extend([{**x, "phase": "full"} for x in h])

    return model, histories, best_f1, best_epoch


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 80) -> float:
    logits = logits.float()
    labels = labels.long()
    log_t = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_t).clamp(0.05, 20.0)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t.detach()).clamp(0.05, 20.0).item())


def softmax_with_temperature(logits: torch.Tensor, temperature: float):
    return torch.softmax(logits.float() / float(temperature), dim=1)


def expected_calibration_error(probs, labels, n_bins: int = 15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def multiclass_brier(probs, labels):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    one_hot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def choose_confidence_threshold(val_probs, target_coverage: float = 0.90):
    conf = np.asarray(val_probs).max(axis=1)
    target_coverage = float(np.clip(target_coverage, 0.01, 1.0))
    return float(np.quantile(conf, 1.0 - target_coverage))


def selective_metrics(probs, labels, threshold):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    keep = conf >= threshold
    if keep.sum() == 0:
        return {"coverage": 0.0, "selective_accuracy": float("nan"), "accepted": 0}
    return {
        "coverage": float(keep.mean()),
        "selective_accuracy": float((pred[keep] == labels[keep]).mean()),
        "accepted": int(keep.sum()),
    }


def detailed_report(labels, probs, class_names):
    preds = np.asarray(probs).argmax(axis=1)
    return classification_report(
        labels, preds, target_names=class_names, digits=4, zero_division=0, output_dict=True
    )


def get_gradcam_target_layer(model: nn.Module, architecture: str):
    a = architecture.lower()
    if a == "custom_cnn":
        # last Conv2d in features
        for layer in reversed(model.features):
            if isinstance(layer, nn.Conv2d):
                return layer
    elif a == "resnet50":
        return model.layer4[-1].conv3
    elif a == "efficientnet_b0":
        return model.features[-1]
    elif a == "densenet121":
        return model.features.denseblock4
    raise ValueError(f"No Grad-CAM target configured for {architecture}")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        acts = self.activations
        grads = self.gradients
        if isinstance(acts, (tuple, list)):
            acts = acts[0]
        if isinstance(grads, (tuple, list)):
            grads = grads[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach()

    def close(self):
        self.h1.remove()
        self.h2.remove()


def preprocess_pil(img: Image.Image, image_size: int = 224):
    _, eval_tfms = get_transforms(image_size)
    return eval_tfms(img.convert("RGB")).unsqueeze(0)


def export_bundle(path, architecture, model, class_names, cfg, temperature, threshold):
    payload = {
        "architecture": architecture,
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": cfg.image_size,
        "temperature": float(temperature),
        "confidence_threshold": float(threshold),
        "title": "An Explainable and Confidence-Calibrated Deep Learning Framework for Reliable Brain Tumor Classification from MRI",
        "split_protocol": "stratified 70:15:15; original Training and Testing folders pooled before splitting",
    }
    torch.save(payload, path)


def load_bundle(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    model = build_model(
        payload["architecture"],
        len(payload["class_names"]),
        pretrained=False
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


from torchvision import transforms

def generate_lime_explanation(model, image_tensor, class_idx, device):
    """Generates a LIME explanation for a given image tensor."""
    # 1. De-normalize the image from PyTorch back to standard 0-1 range for LIME
    img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    explainer = lime_image.LimeImageExplainer()
    normalize = transforms.Normalize(mean.tolist(), std.tolist())

    def batch_predict(images):
        """Wrapper to allow LIME to pass multiple perturbed images to PyTorch."""
        model.eval()
        # Convert LIME's (N, H, W, C) to PyTorch's (N, C, H, W)
        tensor_imgs = torch.tensor(images).permute(0, 3, 1, 2).float()
        # 2. Re-normalize the perturbed batch before feeding it to the model
        tensor_imgs = normalize(tensor_imgs).to(device)
        
        with torch.no_grad():
            logits = model(tensor_imgs)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # Generate the explanation
    explanation = explainer.explain_instance(
        img_np.astype('double'), 
        batch_predict, 
        top_labels=1, 
        hide_color=0, 
        num_samples=500 
    )

    # Get the image and mask for the predicted class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    # Apply the mask boundaries to the image
    lime_img = mark_boundaries(temp, mask)
    
    # Filter out superpixel boundary lines from the dark background
    gray = np.mean(img_np, axis=2)
    background_mask = gray < 0.10
    lime_img[background_mask] = img_np[background_mask]

    return lime_img