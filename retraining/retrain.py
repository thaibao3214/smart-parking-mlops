"""Fine-tune một trong ba mô hình Smart Parking v1 thành candidate v2.

python retraining/retrain.py --model mobilenetv3 --output-dir outputs/mobilenetv3/run_001
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

from evaluate import (ROOT, CHECKPOINTS, TRANSFORM, CocoCropDataset,
                      load_model, evaluate_model, create_validation_loader)


class ConfirmedOnlineDataset(Dataset):
    def __init__(self, confirmed_dir):
        self.root = Path(confirmed_dir)
        self.samples = []
        self.source_images = set()
        self.corrections = 0
        self.label_counts = Counter()
        label_files = sorted(self.root.glob("*/labels.json"))
        if not label_files:
            raise RuntimeError(f"No confirmed labels in {self.root}")
        for label_path in label_files:
            data = json.loads(label_path.read_text(encoding="utf-8"))
            source = data.get("source_image")
            if not isinstance(source, str) or not source:
                raise ValueError(f"Missing source_image: {label_path}")
            self.source_images.add(Path(source).name)
            slots = data.get("slots", [])
            if not slots:
                raise ValueError(f"Empty slots: {label_path}")
            seen_names = set()
            for slot in slots:
                label = slot.get("confirmed_label")
                if label not in ("Empty", "Occupied"):
                    raise ValueError(f"Invalid confirmed_label in {label_path}")
                expected = 0 if label == "Empty" else 1
                if slot.get("label_id") != expected:
                    raise ValueError(f"label_id mismatch in {label_path}")
                name = slot.get("crop_file")
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError(f"Unsafe crop filename in {label_path}")
                if name in seen_names:
                    raise ValueError(f"Duplicate crop filename in {label_path}: {name}")
                seen_names.add(name)
                crop_path = label_path.parent / name
                if not crop_path.is_file():
                    raise FileNotFoundError(crop_path)
                if slot.get("predicted_label") != label:
                    self.corrections += 1
                self.label_counts[label] += 1
                self.samples.append((crop_path, expected))
        if not self.samples:
            raise RuntimeError("Empty confirmed dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            rgb = source.convert("RGB")
        return TRANSFORM(rgb), label


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        batch_n = len(labels)
        loss_sum += loss.item() * batch_n
        correct += (output.argmax(dim=1) == labels).sum().item()
        total += batch_n
    if not total:
        raise RuntimeError("No training batches")
    return {"loss": loss_sum / total, "accuracy": correct / total, "samples_seen": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=CHECKPOINTS, default="mobilenetv3")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--confirmed-dir", type=Path,
                        default=ROOT / "data/confirmed_online/frames")
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/train")
    parser.add_argument("--valid-dir", type=Path, default=ROOT / "data/valid")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--replay-samples", type=int, default=4000)
    parser.add_argument("--eval-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-f1-drop", type=float, default=0.01)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0 or args.replay_samples < 1 or args.eval_samples < 2 or not 0 <= args.max_f1_drop <= 1:
        parser.error("Invalid args: epochs>=1, batch-size>=2, replay>=1, eval>=2, lr>0, 0<=max-f1-drop<=1")

    checkpoint = args.checkpoint or CHECKPOINTS[args.model]
    output_dir = args.output_dir or ROOT / "outputs/retraining" / args.model / "run_001"
    checkpoint_out = output_dir / "candidate_v2.pth"
    report_out = output_dir / "report.json"
    # Không train nếu đích ghi có thể ghi đè artifacts cũ.
    if checkpoint_out.exists() or report_out.exists():
        raise FileExistsError(f"Output already exists: {output_dir}; choose another --output-dir")
    if args.mlflow and not os.environ.get("MLFLOW_TRACKING_URI"):
        raise RuntimeError("Set MLFLOW_TRACKING_URI before --mlflow")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model} | Device: {device}")
    online = ConfirmedOnlineDataset(args.confirmed_dir)
    replay = CocoCropDataset(args.train_dir, max_samples=args.replay_samples,
                             seed=args.seed, exclude_names=online.source_images)
    train_dataset = ConcatDataset([online, replay])
    # Custom CNN dùng BatchNorm1d: batch cuối không được chứa chỉ 1 mẫu.
    drop_last = (len(train_dataset) % args.batch_size == 1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=drop_last,
                              pin_memory=device.type == "cuda",
                              generator=torch.Generator().manual_seed(args.seed))
    valid_loader = create_validation_loader(
        args.valid_dir, max_samples=args.eval_samples,
        batch_size=args.batch_size, seed=args.seed,
        exclude_names=online.source_images,
    )
    print(f"Online={len(online)} replay={len(replay)} validation={len(valid_loader.dataset)}")
    print(f"Confirmed: {dict(online.label_counts)}; corrections={online.corrections}")
    if online.corrections == 0:
        print("WARNING: no human corrections; ensure labels were actually reviewed")
    print("WARNING: evaluation on existing PKLot validation split is not an independent cross-camera test")

    model = load_model(args.model, checkpoint, device)
    baseline = evaluate_model(model, valid_loader, device)
    print("Baseline:", baseline)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, valid_loader, device)
        record = {"epoch": epoch, "train_loss": train_metrics["loss"],
                  "train_accuracy": train_metrics["accuracy"],
                  "val_accuracy": val_metrics["accuracy"],
                  "val_macro_f1": val_metrics["macro_f1"]}
        history.append(record)
        print(f"Epoch {epoch}/{args.epochs}: {record}")

    candidate = evaluate_model(model, valid_loader, device)
    f1_drop = baseline["macro_f1"] - candidate["macro_f1"]
    acc_drop = baseline["accuracy"] - candidate["accuracy"]
    gate = (f1_drop <= args.max_f1_drop and acc_drop <= args.max_f1_drop)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "source_model": "v1",
        "candidate_model": "v2",
        "git_commit": git_commit(),
        "data_dvc_sha256": sha256_file(ROOT / "data.dvc") if (ROOT / "data.dvc").is_file() else None,
        "baseline_checkpoint_sha256": sha256_file(checkpoint),
        "online_frames": len(online.source_images),
        "online_crops": len(online),
        "online_confirmed_distribution": dict(online.label_counts),
        "human_corrections": online.corrections,
        "replay_crops": len(replay),
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
            "passed": gate,
        },
        "deployment_approved": False,
        "notes": "The online source images come from PKLot test and must not be used as independent v2 test data. Validation split may contain same-day/camera near-duplicates.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), checkpoint_out)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.mlflow:
        import mlflow
        mlflow.set_experiment("smart-parking-online-retraining")
        with mlflow.start_run(run_name=f"{args.model}-v2-candidate"):
            mlflow.log_params({
                "model": args.model, "epochs": args.epochs, "lr": args.lr,
                "online_crops": len(online), "replay_crops": len(replay),
                "git_commit": report["git_commit"],
                "data_dvc_sha256": report["data_dvc_sha256"] or "unknown",
            })
            mlflow.log_metrics({
                "v1_accuracy": baseline["accuracy"],
                "v1_macro_f1": baseline["macro_f1"],
                "v2_accuracy": candidate["accuracy"],
                "v2_macro_f1": candidate["macro_f1"],
                "evaluation_gate_passed": int(gate),
            })
            mlflow.log_artifact(str(report_out))
            mlflow.log_artifact(str(checkpoint_out))

    print("\n=== RETRAINING SUMMARY ===")
    print(f"Model: {args.model}")
    print(f"v1 Macro-F1: {baseline['macro_f1']:.5f}")
    print(f"v2 Macro-F1: {candidate['macro_f1']:.5f}")
    print(f"Evaluation Gate: {'PASS' if gate else 'FAIL'}")
    print("Deployment approved: False (manual review required)")
    print("Checkpoint:", checkpoint_out)
    print("Report:", report_out)


if __name__ == "__main__":
    main()
