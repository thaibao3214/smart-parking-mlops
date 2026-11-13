
"""
Smart Parking MLOps - Stage 2 GPU Training Smoke

Train: 128 PKLot training crops
Valid: 64 PKLot validation crops
Epochs: 1
Model: MobileNetV3-Small

No PKLot test crops.
No online crops.
No Production model updates.
"""

import json
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


EXPECTED_DATASET_NAME = "__STAGE2_DATASET_NAME__"
EXPECTED_RUN_ID = "__EXPECTED_RUN_ID__"

INPUT_ROOT = Path("/kaggle/input")

SOURCE = (
    INPUT_ROOT
    / "datasets"
    / "nguynngthibo"
    / EXPECTED_DATASET_NAME
)

WORKDIR = Path("/kaggle/working/stage2_crops")
OUTPUT = Path("/kaggle/working")


class ParkingCrops(Dataset):
    def __init__(self, root, records):
        self.root = Path(root)
        self.records = records

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        path = self.root / record["file"]

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing crop image: {path}"
            )

        with Image.open(path) as image:
            image = self.transform(
                image.convert("RGB")
            )

        label = int(record["label"])

        if label not in (0, 1):
            raise ValueError(
                f"Invalid class label: {label}"
            )

        return image, label


def find_current_dataset():
    print("STAGE2_BEGIN", flush=True)
    print(
        "EXPECTED_DATASET_NAME:",
        EXPECTED_DATASET_NAME,
        flush=True
    )
    print(
        "EXPECTED_RUN_ID:",
        EXPECTED_RUN_ID,
        flush=True
    )
    print(
        "EXPECTED_INPUT_PATH:",
        SOURCE,
        flush=True
    )

    manifests = []

    if INPUT_ROOT.is_dir():
        manifests = sorted(
            INPUT_ROOT.rglob("manifest.json")
        )

    print(
        "AVAILABLE_MANIFESTS:",
        [str(p) for p in manifests[:30]],
        flush=True
    )

    candidates = [
        SOURCE / "manifest.json",
        (
            INPUT_ROOT
            / EXPECTED_DATASET_NAME
            / "manifest.json"
        ),
    ]

    candidates.extend(manifests)

    seen = set()

    for manifest_path in candidates:
        key = str(manifest_path)

        if key in seen:
            continue

        seen.add(key)

        if not manifest_path.is_file():
            continue

        try:
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except (ValueError, OSError):
            continue

        if manifest.get("stage") != "stage2-smoke":
            continue

        if str(
            manifest.get("github_run_id")
        ) != EXPECTED_RUN_ID:
            continue

        # Never accept an older or unrelated dataset.
        if EXPECTED_DATASET_NAME not in str(
            manifest_path.parent
        ):
            continue

        print(
            "ACTUAL_DATASET_PATH:",
            manifest_path.parent,
            flush=True
        )

        print(
            "STAGE2_CURRENT_RUN_DATASET_PASS",
            flush=True
        )

        return manifest_path.parent, manifest

    raise RuntimeError(
        "Current GitHub run dataset is not mounted. "
        "Check AVAILABLE_MANIFESTS and Kaggle "
        "Dataset attachment."
    )


def prepare_data():
    source, manifest = find_current_dataset()

    assert manifest["online_crops_included"] == 0
    assert manifest["test_crops_included"] == 0

    assert len(manifest["train"]) == 128
    assert len(manifest["valid"]) == 64

    print("STAGE2_MANIFEST_PASS", flush=True)

    records = (
        manifest["train"] + manifest["valid"]
    )

    names = [
        record["file"]
        for record in records
    ]

    expected = set(names)

    assert len(expected) == 192
    assert len(names) == 192

    for name in expected:
        path = Path(name)

        assert not path.is_absolute()
        assert ".." not in path.parts
        assert path.parts[0] in ("train", "valid")

    # Kaggle may automatically extract uploaded ZIP files.
    # Support both extracted crops and original ZIP.
    candidates = [
        source,
        source / "crops",
        source / "crops.zip",
    ]

    data_root = None

    for base in candidates:
        if not base.is_dir():
            continue

        if all(
            (base / name).is_file()
            for name in expected
        ):
            data_root = base
            break

    if data_root is None:
        archive = source / "crops.zip"

        if (
            archive.is_file()
            and zipfile.is_zipfile(archive)
        ):
            with zipfile.ZipFile(archive) as z:
                members = set(z.namelist())

                assert members == expected, (
                    "ZIP contents differ from manifest"
                )

                assert z.testzip() is None, (
                    "Corrupted ZIP archive"
                )

                WORKDIR.mkdir(
                    parents=True,
                    exist_ok=True
                )

                z.extractall(WORKDIR)

            data_root = WORKDIR

    if data_root is None:
        visible = sorted(
            str(path.relative_to(source))
            for path in source.rglob("*")
            if path.is_file()
        )

        print(
            "DATASET_VISIBLE_FILES:",
            visible[:30],
            flush=True
        )

        raise RuntimeError(
            "Missing crop images after Kaggle "
            "Dataset processing"
        )

    print(
        "STAGE2_DATA_ROOT:",
        data_root,
        flush=True
    )

    print(
        "STAGE2_CROPS_FOUND_PASS:",
        len(expected),
        flush=True
    )

    return manifest, data_root


def train_one_epoch(manifest, data_root):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Kaggle CUDA GPU is unavailable"
        )

    torch.manual_seed(42)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    print("CUDA_AVAILABLE: True", flush=True)
    print("GPU:", gpu_name, flush=True)

    train_dataset = ParkingCrops(
        data_root,
        manifest["train"]
    )

    valid_dataset = ParkingCrops(
        data_root,
        manifest["valid"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model = mobilenet_v3_small(
        weights=None
    )

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        2
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    model.train()

    losses = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(images)

        loss = criterion(
            predictions,
            labels
        )

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    average_loss = (
        sum(losses) / len(losses)
    )

    print(
        "STAGE2_TRAIN_EPOCH_PASS",
        average_loss,
        flush=True
    )

    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in valid_loader:
            predictions = model(
                images.to(device)
            ).argmax(dim=1).cpu()

            correct += int(
                (predictions == labels).sum()
            )

            total += len(labels)

    accuracy = correct / total

    print(
        "STAGE2_VALIDATION_PASS",
        accuracy,
        flush=True
    )

    return {
        "gpu": gpu_name,
        "train_loss": average_loss,
        "valid_accuracy": accuracy,
    }


def main():
    manifest, data_root = prepare_data()

    metrics = train_one_epoch(
        manifest,
        data_root
    )

    report = {
        "stage": "stage2-smoke",
        "model": "mobilenet_v3_small",
        "weights": "random_initialization",
        "epochs": 1,
        "production_approved": False,
        "train_samples": 128,
        "valid_samples": 64,
        "online_crops_included": 0,
        "test_crops_included": 0,
        "git_commit": manifest["git_commit"],
        "github_run_id": manifest["github_run_id"],
        "data_dvc_sha256": manifest[
            "data_dvc_sha256"
        ],
        **metrics,
    }

    report_path = (
        OUTPUT / "stage2_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "STAGE2_GPU_TRAIN_PASS",
        flush=True
    )

    print(
        json.dumps(
            report,
            indent=2
        ),
        flush=True
    )


if __name__ == "__main__":
    main()
