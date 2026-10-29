"""Train the PyTorch bone-fracture CNN using the existing dataset splits.

Run from the project root with: python -m training.bone_fracture_detection.train_model

The best validation checkpoint is saved to the planned path:
models/bone_fracture_detection/fracture_cnn.keras
The file contains a PyTorch state-dictionary checkpoint despite its retained
`.keras` filename; inference loads it with PyTorch.
"""

from __future__ import annotations

from pathlib import Path
import random
from time import monotonic

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, default_collate

from src.bone_fracture_detection.config import (
    XRAY_ARCHITECTURE_VERSION,
    XRAY_BATCH_SIZE,
    XRAY_CLASS_NAMES,
    XRAY_DATASET_DIR,
    XRAY_EPOCHS,
    XRAY_IMAGE_SIZE,
    XRAY_MIN_FRACTURE_SENSITIVITY,
    XRAY_MODEL_DIR,
    XRAY_MODEL_PATH,
    XRAY_PREPROCESSING_VERSION,
    XRAY_SKIPPED_IMAGES_PATH,
    XRAY_SEED,
)
from src.bone_fracture_detection.model import build_fracture_cnn
from src.bone_fracture_detection.preprocessing import preprocess_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SKIPPED_IMAGE_PATHS: set[str] = set()
SKIPPED_BY_SPLIT: dict[str, int] = {}
IMAGE_TOTALS_BY_SPLIT: dict[str, int] = {}


def initialize_skipped_images_report() -> None:
    """Create a fresh report before processing any dataset images."""
    SKIPPED_IMAGE_PATHS.clear()
    SKIPPED_BY_SPLIT.clear()
    IMAGE_TOTALS_BY_SPLIT.clear()
    XRAY_SKIPPED_IMAGES_PATH.write_text("Unreadable images skipped during training\n", encoding="utf-8")


def record_skipped_image(path: Path, error: Exception, split_name: str) -> None:
    """Log each unreadable image once and persist its path immediately."""
    image_path = str(path.resolve())
    if image_path in SKIPPED_IMAGE_PATHS:
        return

    SKIPPED_IMAGE_PATHS.add(image_path)
    SKIPPED_BY_SPLIT[split_name] = SKIPPED_BY_SPLIT.get(split_name, 0) + 1
    with XRAY_SKIPPED_IMAGES_PATH.open("a", encoding="utf-8") as report:
        report.write(f"{image_path}\n")


class FractureImageDataset(Dataset):
    """Read labeled images from the dataset's class-folder structure."""

    def __init__(self, split: str, *, augment: bool = False) -> None:
        self.split_dir = XRAY_DATASET_DIR / split
        self.split = split
        self.augment = augment
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {self.split_dir}")

        self.samples: list[tuple[Path, int]] = []
        for class_index, class_name in enumerate(XRAY_CLASS_NAMES):
            class_dir = self.split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Dataset class folder not found: {class_dir}")
            self.samples.extend(
                (path, class_index)
                for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )

        if not self.samples:
            raise ValueError(f"No supported images were found in {self.split_dir}")

        IMAGE_TOTALS_BY_SPLIT[split] = len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | None:
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image_tensor = preprocess_image(image, augment=self.augment)
            return image_tensor, label
        except (OSError, ValueError) as exc:
            record_skipped_image(path, exc, self.split)
            return None


def collate_readable_samples(samples):
    """Drop unreadable files from a batch while retaining valid samples."""
    readable_samples = [sample for sample in samples if sample is not None]
    if not readable_samples:
        return None
    return default_collate(readable_samples)


def make_loader(split: str, *, shuffle: bool, device: torch.device) -> DataLoader:
    dataset = FractureImageDataset(split, augment=shuffle)
    return DataLoader(
        dataset,
        batch_size=XRAY_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_readable_samples,
    )


def format_duration(seconds: float) -> str:
    """Format an estimated duration as hours/minutes/seconds."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def print_progress(label: str, completed: int, total: int, started_at: float) -> None:
    """Print an updating batch progress bar and estimated remaining time."""
    total = max(total, 1)
    fraction = min(completed / total, 1.0)
    bar_width = 28
    filled = int(bar_width * fraction)
    elapsed = monotonic() - started_at
    eta = elapsed / completed * max(total - completed, 0) if completed else 0
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
        f"\r{label} [{bar}] {completed}/{total} batches "
        f"({fraction:.0%}) | elapsed {format_duration(elapsed)} "
        f"| left ~{format_duration(eta)}",
        end="",
        flush=True,
    )
    if completed >= total:
        print(flush=True)


def write_skipped_images_summary() -> None:
    """Finalize the skipped-image report without printing skip details."""
    report_lines = ["Unreadable images skipped during training", ""]
    for split in ("train", "val", "test"):
        total = IMAGE_TOTALS_BY_SPLIT.get(split, 0)
        skipped = SKIPPED_BY_SPLIT.get(split, 0)
        rate = skipped / total if total else 0.0
        report_lines.append(f"{split}: {skipped}/{total} skipped ({rate:.2%})")

    report_lines.extend(["", "Skipped image paths:"])
    if SKIPPED_IMAGE_PATHS:
        report_lines.extend(sorted(SKIPPED_IMAGE_PATHS))
    else:
        report_lines.append("None")
    XRAY_SKIPPED_IMAGES_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    progress_label: str = "Evaluation",
    fracture_threshold: float = 0.5,
) -> tuple[float, float]:
    """Run one training or evaluation epoch and return loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    started_at = monotonic()
    total_batches = len(loader)

    for batch_index, batch in enumerate(loader, start=1):
        print_progress(progress_label, batch_index, total_batches, started_at)
        if batch is None:
            continue
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        fracture_index = XRAY_CLASS_NAMES.index("fractured")
        other_index = 1 - fracture_index
        predictions = torch.where(
            torch.softmax(logits, dim=1)[:, fracture_index] >= fracture_threshold,
            fracture_index,
            other_index,
        )
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No readable images were available in this dataset split.")

    return total_loss / total_samples, total_correct / total_samples


def tune_fracture_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Prefer specificity while preserving validation fracture sensitivity."""
    model.eval()
    fracture_index = XRAY_CLASS_NAMES.index("fractured")
    other_index = 1 - fracture_index
    probability_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            images, labels = batch
            logits = model(images.to(device, non_blocking=True))
            probability_batches.append(
                torch.softmax(logits, dim=1)[:, fracture_index].cpu()
            )
            label_batches.append(labels.cpu())

    if not probability_batches:
        raise RuntimeError("No readable validation images were available for threshold tuning.")

    probabilities = torch.cat(probability_batches)
    labels = torch.cat(label_batches)
    candidates = torch.unique(
        torch.cat([torch.tensor([0.0, 0.5, 1.0]), probabilities])
    ).tolist()
    operating_points: list[tuple[float, float, float, float]] = []

    for threshold in candidates:
        predicted_fracture = probabilities >= threshold
        actual_fracture = labels == fracture_index
        true_positive = int((predicted_fracture & actual_fracture).sum())
        true_negative = int(
            ((~predicted_fracture) & (labels == other_index)).sum()
        )
        positives = int(actual_fracture.sum())
        negatives = int((labels == other_index).sum())
        sensitivity = true_positive / positives if positives else 0.0
        specificity = true_negative / negatives if negatives else 0.0
        balanced_accuracy = (sensitivity + specificity) / 2.0
        operating_points.append(
            (float(threshold), sensitivity, specificity, balanced_accuracy)
        )

    feasible = [
        point
        for point in operating_points
        if point[1] >= XRAY_MIN_FRACTURE_SENSITIVITY
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda point: (
                point[2],
                point[1],
                -abs(point[0] - 0.5),
            ),
        )
    else:
        selected = max(
            operating_points,
            key=lambda point: (
                point[3],
                point[2],
                -abs(point[0] - 0.5),
            ),
        )

    threshold, sensitivity, specificity, _ = selected
    return threshold, sensitivity, specificity


def save_checkpoint(model: nn.Module, validation_loss: float) -> None:
    """Save model weights and the label order at the planned checkpoint path."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": list(XRAY_CLASS_NAMES),
        "image_size": list(XRAY_IMAGE_SIZE),
        "architecture_version": XRAY_ARCHITECTURE_VERSION,
        "preprocessing_version": XRAY_PREPROCESSING_VERSION,
        "validation_loss": validation_loss,
        "fracture_threshold": 0.5,
    }
    torch.save(checkpoint, XRAY_MODEL_PATH)


def main() -> None:
    """Train on train/, select the best val/ checkpoint, and evaluate test/."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot access a CUDA GPU, so training was stopped before "
            "loading the dataset. Install the CUDA-enabled PyTorch build for "
            "Windows from https://pytorch.org/get-started/locally/ and retry."
        )

    torch.manual_seed(XRAY_SEED)
    torch.cuda.manual_seed_all(XRAY_SEED)
    random.seed(XRAY_SEED)
    device = torch.device("cuda")
    print(f"Training on: {torch.cuda.get_device_name(0)}")

    XRAY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    initialize_skipped_images_report()
    train_loader = make_loader("train", shuffle=True, device=device)
    validation_loader = make_loader("val", shuffle=False, device=device)
    test_loader = make_loader("test", shuffle=False, device=device)

    model = build_fracture_cnn().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    best_validation_accuracy = 0.0
    epochs_without_accuracy_improvement = 0
    loss_patience = 6
    accuracy_patience = 8
    accuracy_min_delta = 0.001

    for epoch in range(1, XRAY_EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            progress_label=f"Epoch {epoch}/{XRAY_EPOCHS} train",
        )
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            progress_label=f"Epoch {epoch}/{XRAY_EPOCHS} val",
        )
        print(
            f"Epoch {epoch}/{XRAY_EPOCHS} | "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.4f} | "
            f"val loss {validation_loss:.4f}, accuracy {validation_accuracy:.4f}"
        )
        scheduler.step(validation_loss)

        if validation_accuracy > best_validation_accuracy + accuracy_min_delta:
            best_validation_accuracy = validation_accuracy
            epochs_without_accuracy_improvement = 0
        else:
            epochs_without_accuracy_improvement += 1

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            save_checkpoint(model, validation_loss)
            print(f"Saved best checkpoint to {XRAY_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= loss_patience:
                print(
                    "Early stopping: validation loss stalled for "
                    f"{loss_patience} epochs."
                )
                break

        if epochs_without_accuracy_improvement >= accuracy_patience:
            print(
                "Early stopping: validation accuracy did not improve by "
                f"{accuracy_min_delta:.3f} for {accuracy_patience} epochs."
            )
            break

    checkpoint = torch.load(XRAY_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    (
        fracture_threshold,
        validation_sensitivity,
        validation_specificity,
    ) = tune_fracture_threshold(model, validation_loader, device)
    checkpoint["fracture_threshold"] = fracture_threshold
    checkpoint["validation_sensitivity_at_threshold"] = validation_sensitivity
    checkpoint["validation_specificity_at_threshold"] = validation_specificity
    torch.save(checkpoint, XRAY_MODEL_PATH)
    print(
        f"Validation threshold: {fracture_threshold:.4f} | "
        f"sensitivity {validation_sensitivity:.4f} | "
        f"specificity {validation_specificity:.4f}"
    )
    test_loss, test_accuracy = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        progress_label="Test",
        fracture_threshold=fracture_threshold,
    )
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}")
    print(f"Best model saved to {XRAY_MODEL_PATH}")
    write_skipped_images_summary()


if __name__ == "__main__":
    main()
