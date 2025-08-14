"""Evaluate trained advertisement models on a test set.

Example:
    python -m general_dl_ads_project.src.evaluate \
        --model_type mlp --test_csv test.csv --target_col clicked \
        --model_path models/mlp_best.pth --preprocessor models/preprocessor.pkl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import functional as F
from torchvision import datasets, transforms

from .data_loader import AdDataset, load_preprocessor
from .model_mlp import MLP
from .model_cnn import AdImageCNN


def _eval_model(model, loader, device, is_binary=True):
    model.eval()
    probs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if is_binary:
                probs = torch.sigmoid(logits.view(-1))
            else:
                probs = F.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
            labels_list.append(y.numpy())
    probs = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    preds = (probs > 0.5).astype(int) if is_binary else probs.argmax(axis=1)
    return preds, probs, labels


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    if args.model_type == "mlp":
        if not args.preprocessor:
            raise ValueError("Preprocessor path is required for MLP evaluation")
        pre = load_preprocessor(args.preprocessor)
        df = pd.read_csv(args.test_csv)
        X = pre.transform(df.drop(columns=[args.target_col]))
        y = df[args.target_col].values
        test_ds = AdDataset(X, y)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False
        )
        input_dim = test_ds.features.shape[1]
        model = MLP(input_dim=input_dim)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        preds, probs, labels = _eval_model(
            model, test_loader, device, is_binary=True
        )
    else:
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        dataset = datasets.ImageFolder(args.test_dir, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )
        model = AdImageCNN(
            model_name=args.backbone,
            num_classes=len(dataset.classes),
            pretrained=False,
        )
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        preds, probs, labels = _eval_model(
            model, test_loader, device, is_binary=len(dataset.classes) == 2
        )

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="binary" if len(np.unique(labels)) == 2 else "macro")),
        "recall": float(recall_score(labels, preds, average="binary" if len(np.unique(labels)) == 2 else "macro")),
        "f1": float(f1_score(labels, preds, average="binary" if len(np.unique(labels)) == 2 else "macro")),
    }
    try:
        if preds.ndim == 1 and probs.ndim == 1:
            metrics["auc"] = float(roc_auc_score(labels, probs))
        else:
            metrics["auc"] = float(
                roc_auc_score(labels, probs, multi_class="ovr")
            )
    except ValueError:
        metrics["auc"] = float("nan")

    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(cm.shape[0]))
    ax.set_yticks(range(cm.shape[0]))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "confusion_matrix.png"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate advertisement models")
    parser.add_argument("--model_type", choices=["mlp", "cnn"], required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_path", type=str, required=True)

    # MLP args
    parser.add_argument("--test_csv", type=str)
    parser.add_argument("--target_col", type=str, default="clicked")
    parser.add_argument("--preprocessor", type=str, default=None)

    # CNN args
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--backbone", type=str, default="resnet18")

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
