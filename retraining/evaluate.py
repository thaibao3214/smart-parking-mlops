"""Đánh giá các checkpoint Smart Parking trên PKLot COCO.

Chạy: python retraining/evaluate.py --model resnet18 --max-samples 300
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, resnet18
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "mobilenetv3": ROOT / "mobilenet_model/mobilenet_v3_small_best.pth",
    "resnet18": ROOT / "resnet_model/resnet18_best.pth",
    "custom_cnn": ROOT / "cnn_model/optimized_custom_cnn_best.pth",
}
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class OptimizedPKLotCNN(nn.Module):
    """Khớp kiến trúc OptimizedPKLotCNN trong api/main.py."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(self.global_pool(x))


def build_model(model_name):
    if model_name == "mobilenetv3":
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    elif model_name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "custom_cnn":
        model = OptimizedPKLotCNN(2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def load_model(model_name, checkpoint_path, device):
    model = build_model(model_name)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a state_dict or a dict containing one")
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class CocoCropDataset(Dataset):
    """Dùng COCO bbox; mapping category_id 1=>Empty/0, 2=>Occupied/1."""
    def __init__(self, data_dir, max_samples=None, seed=42, exclude_names=None):
        self.data_dir = Path(data_dir)
        with (self.data_dir / "_annotations.coco.json").open(encoding="utf-8") as f:
            coco = json.load(f)
        image_index = {item["id"]: item for item in coco["images"]}
        exclude = {Path(name).name for name in (exclude_names or [])}
        samples = []
        for ann in coco["annotations"]:
            cat = ann.get("category_id")
            if cat not in (1, 2):
                continue
            image = image_index.get(ann["image_id"])
            if image is None:
                continue
            name = Path(image["file_name"]).name
            if name in exclude:
                continue
            path = self.data_dir / image["file_name"]
            if not path.is_file():
                continue
            x, y, w, h = map(float, ann["bbox"])
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                continue
            width = int(image.get("width", 0))
            height = int(image.get("height", 0))
            if width > 0 and height > 0:
                x1, y1 = max(0, math.floor(x)), max(0, math.floor(y))
                x2, y2 = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
                if x2 <= x1 or y2 <= y1:
                    continue
            samples.append((path, (x, y, w, h), cat - 1))
        random.Random(seed).shuffle(samples)
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be >= 1")
            samples = samples[:max_samples]
        if not samples:
            raise RuntimeError(f"No valid COCO crops in {self.data_dir}")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, (x, y, w, h), label = self.samples[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        x1, y1 = max(0, math.floor(x)), max(0, math.floor(y))
        x2, y2 = min(rgb.width, math.ceil(x + w)), min(rgb.height, math.ceil(y + h))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox for {path.name}")
        return TRANSFORM(rgb.crop((x1, y1, x2, y2))), label


def create_validation_loader(valid_dir, max_samples=3000, batch_size=64,
                             seed=42, exclude_names=None):
    dataset = CocoCropDataset(valid_dir, max_samples=max_samples,
                              seed=seed, exclude_names=exclude_names)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_model(model, loader, device):
    model.eval()
    confusion = [[0, 0], [0, 0]]
    with torch.inference_mode():
        for images, labels in loader:
            pred = model(images.to(device)).argmax(dim=1).cpu().tolist()
            for truth, prediction in zip(labels.tolist(), pred):
                confusion[int(truth)][int(prediction)] += 1
    n = sum(map(sum, confusion))
    if not n:
        raise RuntimeError("Empty evaluation set")
    precisions, recalls, f1s = [], [], []
    for cls in (0, 1):
        tp = confusion[cls][cls]
        fp = sum(confusion[r][cls] for r in (0, 1)) - tp
        fn = sum(confusion[cls]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "samples": n,
        "accuracy": (confusion[0][0] + confusion[1][1]) / n,
        "macro_precision": sum(precisions) / 2,
        "macro_recall": sum(recalls) / 2,
        "macro_f1": sum(f1s) / 2,
        "confusion_matrix": confusion,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=CHECKPOINTS, default="mobilenetv3")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--valid-dir", type=Path, default=ROOT / "data/valid")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or CHECKPOINTS[args.model]
    model = load_model(args.model, checkpoint, device)
    loader = create_validation_loader(args.valid_dir, args.max_samples,
                                      args.batch_size, args.seed)
    print(json.dumps({"model": args.model, "checkpoint": str(checkpoint),
                      "metrics": evaluate_model(model, loader, device)}, indent=2))


if __name__ == "__main__":
    main()
