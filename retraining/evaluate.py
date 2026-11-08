
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def load_model(checkpoint_path, device):
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, 2
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


class CocoCropDataset(Dataset):
    """
    category_id=1: Empty -> 0
    category_id=2: Occupied -> 1

    Chỉ sử dụng COCO bbox và nhãn cho việc
    chuẩn bị training/evaluation.
    """

    def __init__(
        self,
        data_dir,
        max_samples=None,
        seed=42,
        exclude_names=None
    ):
        self.data_dir = Path(data_dir)
        json_path = self.data_dir / "_annotations.coco.json"

        with json_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        images = {
            item["id"]: item
            for item in coco["images"]
        }

        exclude_names = set(exclude_names or [])
        samples = []

        for ann in coco["annotations"]:
            category = ann.get("category_id")

            if category not in (1, 2):
                continue

            image = images.get(ann["image_id"])
            if image is None:
                continue

            name = Path(image["file_name"]).name

            if name in exclude_names:
                continue

            x, y, w, h = ann["bbox"]

            if w <= 0 or h <= 0:
                continue

            path = self.data_dir / image["file_name"]

            if not path.is_file():
                continue

            samples.append((
                path,
                (float(x), float(y), float(w), float(h)),
                category - 1
            ))

        rng = random.Random(seed)
        rng.shuffle(samples)

        if max_samples is not None:
            samples = samples[:max_samples]

        if not samples:
            raise RuntimeError(
                f"Khong co sample hop le: {data_dir}"
            )

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, bbox, label = self.samples[index]

        with Image.open(path) as source:
            image = source.convert("RGB")

        x, y, w, h = bbox

        x1 = max(0, math.floor(x))
        y1 = max(0, math.floor(y))
        x2 = min(image.width, math.ceil(x + w))
        y2 = min(image.height, math.ceil(y + h))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop: {path}")

        crop = image.crop((x1, y1, x2, y2))

        return TRANSFORM(crop), label


def evaluate_model(model, loader, device):
    model.eval()

    # Ma tran nham lan:
    # confusion[ground_truth][prediction]
    confusion = [[0, 0], [0, 0]]

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            for truth, pred in zip(labels, predictions):
                confusion[int(truth)][int(pred)] += 1

    total = sum(sum(row) for row in confusion)

    if total == 0:
        raise RuntimeError("Evaluation set is empty")

    accuracy = (
        confusion[0][0] + confusion[1][1]
    ) / total

    precisions = []
    recalls = []
    f1_scores = []

    for cls in (0, 1):
        tp = confusion[cls][cls]

        fp = sum(confusion[r][cls] for r in (0, 1)) - tp
        fn = sum(confusion[cls]) - tp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "samples": total,
        "accuracy": accuracy,
        "macro_precision": sum(precisions) / 2,
        "macro_recall": sum(recalls) / 2,
        "macro_f1": sum(f1_scores) / 2,
        "confusion_matrix": confusion
    }


def create_validation_loader(
    valid_dir,
    max_samples=3000,
    batch_size=64,
    seed=42,
    exclude_names=None
):
    dataset = CocoCropDataset(
        valid_dir,
        max_samples=max_samples,
        seed=seed,
        exclude_names=exclude_names
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT / "mobilenet_model"
            / "mobilenet_v3_small_best.pth"
        )
    )
    parser.add_argument(
        "--valid-dir",
        default=str(ROOT / "data" / "valid")
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(args.checkpoint, device)

    loader = create_validation_loader(
        args.valid_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )

    metrics = evaluate_model(model, loader, device)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
