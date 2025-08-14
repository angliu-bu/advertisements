"""Training script for advertisement models.

Example:
    # Train an MLP on tabular data
    python -m general_dl_ads_project.src.train \
        --model_type mlp --train_csv train.csv --val_csv val.csv \
        --target_col clicked --epochs 10

    # Train a CNN on image data
    python -m general_dl_ads_project.src.train \
        --model_type cnn --train_dir images/train --val_dir images/val \
        --epochs 5 --backbone resnet18
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from .data_loader import load_tabular_data, save_preprocessor
from .model_mlp import MLP
from .model_cnn import AdImageCNN


def _select_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    if name.lower() == "sgd":
        return SGD(params, lr=lr, momentum=0.9)
    return Adam(params, lr=lr)


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        if out.ndim == 1 or out.shape[1] == 1:
            out = out.view(-1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def _evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if out.ndim == 1 or out.shape[1] == 1:
                out = out.view(-1)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.model_dir, exist_ok=True)

    if args.model_type == "mlp":
        loaders, preprocessor = load_tabular_data(
            args.train_csv,
            target_col=args.target_col,
            val_csv=args.val_csv,
            test_csv=None,
            batch_size=args.batch_size,
        )
        input_dim = loaders.train.dataset.features.shape[1]
        model = MLP(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            activation=args.activation,
            dropout=args.dropout,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        save_preprocessor(preprocessor, os.path.join(args.model_dir, "preprocessor.pkl"))
        train_loader, val_loader = loaders.train, loaders.val

    else:  # CNN training
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        train_dataset = datasets.ImageFolder(args.train_dir, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = None
        if args.val_dir:
            val_dataset = datasets.ImageFolder(args.val_dir, transform=transform)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False
            )
        model = AdImageCNN(
            model_name=args.backbone,
            num_classes=len(train_dataset.classes),
            pretrained=True,
            fine_tune=not args.freeze,
        ).to(device)
        criterion = nn.CrossEntropyLoss()

    optimizer = _select_optimizer(args.optimizer, model.parameters(), args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_loss = float("inf")
    patience_counter = 0
    for epoch in range(args.epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _evaluate(model, val_loader, criterion, device) if val_loader else train_loss
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.model_dir, f"{args.model_type}_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

        print(f"Epoch {epoch+1}/{args.epochs} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train advertisement models")
    parser.add_argument("--model_type", choices=["mlp", "cnn"], required=True)

    # Common args
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step_size", type=int, default=5, help="Scheduler step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="Scheduler decay")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--model_dir", type=str, default="models")

    # MLP specific
    parser.add_argument("--train_csv", type=str)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--target_col", type=str, default="clicked")
    parser.add_argument("--hidden_dims", type=int, nargs="*", default=[64, 32])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.0)

    # CNN specific
    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--freeze", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
