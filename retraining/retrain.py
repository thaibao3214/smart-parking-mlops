
import argparse
import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset
)
from PIL import Image

from evaluate import (
    ROOT,
    TRANSFORM,
    CocoCropDataset,
    load_model,
    evaluate_model,
    create_validation_loader
)


class ConfirmedOnlineDataset(Dataset):
    """
    Doc anh crop va nhan da xac nhan.
    Khong dung predicted_label lam nhan training.
    """

    def __init__(self, confirmed_dir):
        self.root = Path(confirmed_dir)
        self.samples = []
        self.source_images = set()
        self.corrections = 0

        label_files = sorted(
            self.root.glob("*/labels.json")
        )

        if not label_files:
            raise RuntimeError(
                "Khong tim thay confirmed labels"
            )

        for label_path in label_files:
            data = json.loads(
                label_path.read_text(encoding="utf-8")
            )

            source = data.get("source_image")

            if not source:
                raise ValueError(
                    f"Missing source_image: {label_path}"
                )

            self.source_images.add(source)

            for slot in data.get("slots", []):
                label = slot.get("confirmed_label")
                label_id = slot.get("label_id")

                if label not in ("Empty", "Occupied"):
                    raise ValueError(
                        f"Invalid label: {label_path}"
                    )

                expected = (
                    0 if label == "Empty" else 1
                )

                if label_id != expected:
                    raise ValueError(
                        f"label_id mismatch: {label_path}"
                    )

                crop_name = slot["crop_file"]
                crop_path = label_path.parent / crop_name

                if not crop_path.is_file():
                    raise FileNotFoundError(crop_path)

                if (
                    slot.get("predicted_label")
                    != label
                ):
                    self.corrections += 1

                self.samples.append(
                    (crop_path, expected)
                )

        if not self.samples:
            raise RuntimeError("Empty confirmed dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as source:
            image = source.convert("RGB")

        return TRANSFORM(image), label


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def train_one_epoch(
    model, loader, optimizer, criterion, device
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total += batch_size

    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total
    }


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
        "--confirmed-dir",
        default=str(
            ROOT / "data" / "confirmed_online" / "frames"
        )
    )
    parser.add_argument(
        "--train-dir",
        default=str(ROOT / "data" / "train")
    )
    parser.add_argument(
        "--valid-dir",
        default=str(ROOT / "data" / "valid")
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "retraining")
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--replay-samples", type=int, default=4000)
    parser.add_argument("--eval-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-f1-drop", type=float, default=0.01)

    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log metrics/artifacts to configured MLflow server"
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.replay_samples < 1
        or args.eval_samples < 1
        or args.lr <= 0
        or not 0 <= args.max_f1_drop <= 1
    ):
        parser.error("Invalid training arguments")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # 1. Doc du lieu online da xac nhan
    online_dataset = ConfirmedOnlineDataset(
        args.confirmed_dir
    )

    print("Online crops:", len(online_dataset))
    print("Source frames:", len(online_dataset.source_images))
    print("Human corrections:", online_dataset.corrections)

    # 2. Replay du lieu PKLot Train cu
    replay_dataset = CocoCropDataset(
        args.train_dir,
        max_samples=args.replay_samples,
        seed=args.seed,
        exclude_names=online_dataset.source_images
    )

    print("Replay crops:", len(replay_dataset))

    train_dataset = ConcatDataset([
        online_dataset,
        replay_dataset
    ])

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # 3. Tap validation khong su dung online crops
    valid_loader = create_validation_loader(
        args.valid_dir,
        max_samples=args.eval_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        exclude_names=online_dataset.source_images
    )

    # 4. Load MobileNetV3 v1
    model = load_model(
        args.checkpoint,
        device
    )

    print("\nEvaluating baseline v1...")
    baseline = evaluate_model(
        model,
        valid_loader,
        device
    )

    print("Baseline:", baseline)

    # 5. Fine-tune toan bo model voi LR nho
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        valid_metrics = evaluate_model(
            model,
            valid_loader,
            device
        )

        epoch_report = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": valid_metrics["accuracy"],
            "val_macro_f1": valid_metrics["macro_f1"]
        }

        history.append(epoch_report)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={train_metrics['loss']:.5f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_acc={valid_metrics['accuracy']:.4f} | "
            f"val_f1={valid_metrics['macro_f1']:.4f}"
        )

    # 6. Danh gia v2 tren CUNG tap validation
    candidate = evaluate_model(
        model,
        valid_loader,
        device
    )

    # Evaluation Gate: khong cho phep suy giam qua nguong
    f1_drop = baseline["macro_f1"] - candidate["macro_f1"]
    acc_drop = baseline["accuracy"] - candidate["accuracy"]

    gate_passed = (
        f1_drop <= args.max_f1_drop
        and acc_drop <= args.max_f1_drop
    )

    # 7. Luu candidate, KHONG overwrite v1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_out = output_dir / "candidate_v2.pth"
    report_out = output_dir / "report.json"

    # Khong ghi de candidate cu
    if checkpoint_out.exists() or report_out.exists():
        raise FileExistsError(
            "Candidate output already exists. "
            "Use a new --output-dir for each run."
        )

    torch.save(
        model.cpu().state_dict(),
        checkpoint_out
    )

    dvc_file = ROOT / "data.dvc"

    report = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_commit": git_commit(),
        "data_dvc_sha256": (
            sha256_file(dvc_file)
            if dvc_file.is_file() else None
        ),
        "baseline_checkpoint_sha256": sha256_file(
            args.checkpoint
        ),
        "source_model": "MobileNetV3-Small v1",
        "candidate_model": "MobileNetV3-Small v2",
        "online_frames": len(online_dataset.source_images),
        "online_crops": len(online_dataset),
        "replay_crops": len(replay_dataset),
        "validation_crops": candidate["samples"],
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "baseline": baseline,
        "candidate": candidate,
        "history": history,
        "evaluation_gate": {
            "max_allowed_drop": args.max_f1_drop,
            "f1_drop": f1_drop,
            "accuracy_drop": acc_drop,
            "passed": gate_passed
        },
        "deployment_approved": False,
        "warning": (
            "Online data originated from PKLot test. "
            "Validation is not an independent "
            "cross-camera generalization test."
        )
    }

    report_out.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # 8. Optional MLflow Tracking
    if args.mlflow:
        import mlflow

        if not os.environ.get("MLFLOW_TRACKING_URI"):
            raise RuntimeError(
                "Set MLFLOW_TRACKING_URI before --mlflow"
            )

        mlflow.set_experiment(
            "smart-parking-online-retraining"
        )

        with mlflow.start_run(
            run_name="mobilenetv3-v2-candidate"
        ):
            mlflow.log_params({
                "model": "mobilenet_v3_small",
                "epochs": args.epochs,
                "lr": args.lr,
                "online_crops": len(online_dataset),
                "replay_crops": len(replay_dataset),
                "git_commit": report["git_commit"],
                "data_dvc_sha256": report["data_dvc_sha256"]
            })

            mlflow.log_metrics({
                "v1_accuracy": baseline["accuracy"],
                "v1_macro_f1": baseline["macro_f1"],
                "v2_accuracy": candidate["accuracy"],
                "v2_macro_f1": candidate["macro_f1"],
                "evaluation_gate_passed": int(gate_passed)
            })

            mlflow.log_artifact(str(report_out))
            mlflow.log_artifact(str(checkpoint_out))

    print("\n========== RETRAINING RESULT ==========")
    print("Baseline v1 F1:", baseline["macro_f1"])
    print("Candidate v2 F1:", candidate["macro_f1"])
    print("Gate passed:", gate_passed)
    print("Checkpoint:", checkpoint_out)
    print("Report:", report_out)
    print("Deployment approved: False")
    print("======================================")

    if not gate_passed:
        print(
            "WARNING: Candidate did not pass the gate. "
            "Do not deploy."
        )


if __name__ == "__main__":
    main()
