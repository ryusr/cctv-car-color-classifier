import argparse
import copy
import csv
import json
import os
import random
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 for CCTV car color classification."
    )
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset root.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="DataLoader workers.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/mobilenetv2_car_color",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience based on validation accuracy.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing for cross entropy.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(img_size: int) -> Dict[str, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return {"train": train_tfms, "val": eval_tfms, "test": eval_tfms}


def build_datasets(data_dir: Path, img_size: int):
    transforms_map = build_transforms(img_size)
    datasets_map = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        datasets_map[split] = datasets.ImageFolder(
            root=str(split_dir),
            transform=transforms_map[split],
        )
    return datasets_map


def build_dataloaders(
    datasets_map: Dict[str, datasets.ImageFolder],
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    pin_memory = device.type == "cuda"
    loaders = {
        "train": DataLoader(
            datasets_map["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(
            datasets_map["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            datasets_map["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
    }
    return loaders


def get_class_weights(train_dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = Counter(train_dataset.targets)
    max_index = len(train_dataset.classes)
    weights = []
    total = sum(counts.values())
    for idx in range(max_index):
        class_count = counts[idx]
        weights.append(total / (max_index * class_count))
    return torch.tensor(weights, dtype=torch.float32)


def create_model(num_classes: int) -> nn.Module:
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    except AttributeError:
        model = models.mobilenet_v2(pretrained=True)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_scaler(device: torch.device, amp_enabled: bool):
    enabled = amp_enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def get_autocast(device: torch.device, amp_enabled: bool):
    enabled = amp_enabled and device.type == "cuda"
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with get_autocast(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

        progress.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{running_correct / total:.4f}",
        )

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    class_correct = Counter()
    class_total = Counter()

    progress = tqdm(loader, desc="Eval ", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with get_autocast(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (preds == targets).sum().item()
        total += batch_size

        for pred, target in zip(preds.cpu().tolist(), targets.cpu().tolist()):
            class_total[target] += 1
            if pred == target:
                class_correct[target] += 1

        progress.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{running_correct / total:.4f}",
        )

    per_class_acc = {
        idx: (class_correct[idx] / class_total[idx]) if class_total[idx] else 0.0
        for idx in class_total
    }
    return running_loss / total, running_correct / total, per_class_acc


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def write_history_csv(history, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "lr",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def print_dataset_summary(dataset_map: Dict[str, datasets.ImageFolder]) -> None:
    for split, dataset in dataset_map.items():
        counts = Counter(dataset.targets)
        class_counts = {
            dataset.classes[idx]: counts[idx] for idx in range(len(dataset.classes))
        }
        print(f"[{split}] samples={len(dataset)} classes={len(dataset.classes)}")
        print(json.dumps(class_counts, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    amp_enabled = not args.no_amp

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA not found, training will run on CPU.")

    dataset_map = build_datasets(data_dir, args.img_size)
    print_dataset_summary(dataset_map)
    dataloaders = build_dataloaders(
        datasets_map=dataset_map,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model = create_model(num_classes=len(dataset_map["train"].classes)).to(device)
    class_weights = get_class_weights(dataset_map["train"]).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = get_scaler(device, amp_enabled)

    class_to_idx_path = output_dir / "class_to_idx.json"
    with class_to_idx_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_map["train"].class_to_idx, f, indent=2)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )
        val_loss, val_acc, _ = evaluate(
            model=model,
            loader=dataloaders["val"],
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={current_lr:.6f} time={epoch_time:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "lr": f"{current_lr:.8f}",
            }
        )

        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "class_to_idx": dataset_map["train"].class_to_idx,
            "args": vars(args),
        }
        save_checkpoint(last_checkpoint, output_dir / "last.pt")

        if best_state is None or val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "class_to_idx": dataset_map["train"].class_to_idx,
                "args": vars(args),
            }
            save_checkpoint(best_checkpoint, output_dir / "best.pt")
            print(f"Saved new best model to {output_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping triggered after {args.patience} "
                    "epochs without validation improvement."
                )
                break

    write_history_csv(history, output_dir / "history.csv")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_acc, val_per_class = evaluate(
        model=model,
        loader=dataloaders["val"],
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )
    test_loss, test_acc, test_per_class = evaluate(
        model=model,
        loader=dataloaders["test"],
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )

    idx_to_class = {idx: name for name, idx in dataset_map["train"].class_to_idx.items()}
    val_per_class_named = {
        idx_to_class[idx]: round(acc, 4) for idx, acc in sorted(val_per_class.items())
    }
    test_per_class_named = {
        idx_to_class[idx]: round(acc, 4) for idx, acc in sorted(test_per_class.items())
    }

    summary = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "final_val_loss": round(val_loss, 4),
        "final_val_acc": round(val_acc, 4),
        "final_test_loss": round(test_loss, 4),
        "final_test_acc": round(test_acc, 4),
        "classes": dataset_map["train"].classes,
        "val_per_class_accuracy": val_per_class_named,
        "test_per_class_accuracy": test_per_class_named,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining finished.")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
