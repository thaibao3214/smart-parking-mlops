
"""
Stage 2 - Smart Parking GPU Training Smoke Test

Data:
- 128 PKLot training crops
- 64 PKLot validation crops
- No PKLot test crops
- No confirmed_online crops

This is a real one-epoch GPU training test.
It does NOT update the Production Model Registry.
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


SOURCE = Path(
    "/kaggle/input/datasets/"
    "nguynngthibo/smart-parking-stage2-smoke"
)

WORKDIR = Path("/kaggle/working/stage2_crops")
OUT = Path("/kaggle/working")


class ParkingCrops(Dataset):
    def __init__(self, root, records):
        self.root = Path(root)
        self.records = records

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        path = self.root / record["file"]

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing training image: {path}"
            )

        with Image.open(path) as image:
            image = self.transform(
                image.convert("RGB")
            )

        label = int(record["label"])

        return image, label


def prepare_data():
    print("STAGE2_BEGIN", flush=True)

    print("INPUT_PATH:", SOURCE, flush=True)

    assert SOURCE.is_dir(), (
        f"Kaggle dataset not mounted: {SOURCE}"
    )

    manifest_path = SOURCE / "manifest.json"
    archive_path = SOURCE / "crops.zip"

    assert manifest_path.is_file(), (
        f"Missing manifest: {manifest_path}"
    )

    assert archive_path.is_file(), (
        f"Missing crop archive: {archive_path}"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["stage"] == "stage2-smoke"

    assert manifest["online_crops_included"] == 0
    assert manifest["test_crops_included"] == 0

    assert len(manifest["train"]) == 128
    assert len(manifest["valid"]) == 64

    print("STAGE2_MANIFEST_PASS", flush=True)

    WORKDIR.mkdir(
        parents=True,
        exist_ok=True
    )

    expected_files = {
        record["file"]
        for split in ("train", "valid")
        for record in manifest[split]
    }

    assert len(expected_files) == 192, (
        "Duplicate crop filenames detected"
    )

    with zipfile.ZipFile(archive_path, "r") as z:
        names = set(z.namelist())

        assert names == expected_files, (
            "ZIP contents do not match manifest"
        )

        assert z.testzip() is None, (
            "Corrupted crops.zip"
        )

        for name in names:
            path = Path(name)

            assert not path.is_absolute()
            assert ".." not in path.parts
            assert path.parts[0] in ("train", "valid")

        z.extractall(WORKDIR)

    for split in ("train", "valid"):
        for record in manifest[split]:
            path = WORKDIR / record["file"]

            assert path.is_file(), (
                f"Extracted crop missing: {path}"
            )

    print(
        "STAGE2_ZIP_EXTRACT_PASS",
        len(expected_files),
        flush=True
    )

    return manifest


def train_one_epoch(manifest):
    assert torch.cuda.is_available(), (
        "Kaggle GPU is unavailable"
    )

    torch.manual_seed(42)

    device = torch.device("cuda")

    gpu_name = torch.cuda.get_device_name(0)

    print("CUDA_AVAILABLE: True", flush=True)
    print("GPU:", gpu_name, flush=True)

    train_dataset = ParkingCrops(
        WORKDIR,
        manifest["train"]
    )

    valid_dataset = ParkingCrops(
        WORKDIR,
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

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    # Real training: 1 epoch
    model.train()

    total_loss = 0.0
    train_batches = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)

        loss = loss_function(
            predictions,
            labels
        )

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        train_batches += 1

    average_loss = total_loss / train_batches

    print(
        "STAGE2_TRAIN_EPOCH_PASS",
        f"loss={average_loss:.6f}",
        flush=True
    )

    # Validation
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in valid_loader:
            images = images.to(device)

            predictions = model(
                images
            ).argmax(dim=1).cpu()

            correct += int(
                (predictions == labels).sum()
            )

            total += len(labels)

    validation_accuracy = correct / total

    print(
        "STAGE2_VALIDATION_PASS",
        f"accuracy={validation_accuracy:.6f}",
        flush=True
    )

    return {
        "gpu": gpu_name,
        "train_loss": average_loss,
        "valid_accuracy": validation_accuracy,
    }


def main():
    manifest = prepare_data()

    metrics = train_one_epoch(manifest)

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
        **metrics
    }

    report_path = OUT / "stage2_report.json"

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
        json.dumps(report, indent=2),
        flush=True
    )


if __name__ == "__main__":
    main()
