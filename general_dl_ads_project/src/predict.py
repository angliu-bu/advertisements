"""Generate predictions using trained advertisement models.

Example:
    python -m general_dl_ads_project.src.predict \
        --model_type mlp --input_csv new_ads.csv \
        --model_path models/mlp_best.pth --preprocessor models/preprocessor.pkl \
        --output_path preds.csv
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

from .data_loader import AdDataset, load_preprocessor
from .model_mlp import MLP
from .model_cnn import AdImageCNN


def predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    if args.model_type == "mlp":
        pre = load_preprocessor(args.preprocessor)
        df = pd.read_csv(args.input_csv)
        X = pre.transform(df)
        dummy_labels = np.zeros(len(df))
        dataset = AdDataset(X, dummy_labels)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )
        model = MLP(input_dim=dataset.features.shape[1])
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        preds: List[float] = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x).view(-1)
                probs = torch.sigmoid(logits)
                preds.extend(probs.cpu().numpy())
        pd.DataFrame({"prediction": preds}).to_csv(args.output_path, index=False)
    else:
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        dataset = datasets.ImageFolder(args.input_dir, transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )
        model = AdImageCNN(
            model_name=args.backbone,
            num_classes=len(dataset.classes),
            pretrained=False,
        )
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                probs = torch.softmax(model(x), dim=1)
                preds.extend(probs.cpu().numpy())
        np.savetxt(args.output_path, np.array(preds), delimiter=",")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions")
    parser.add_argument("--model_type", choices=["mlp", "cnn"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, required=True)

    # MLP args
    parser.add_argument("--input_csv", type=str)
    parser.add_argument("--preprocessor", type=str, default=None)

    # CNN args
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--backbone", type=str, default="resnet18")

    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
