#!/usr/bin/env python3
"""Smart Parking MLOps project audit (read-only, standard library only)."""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

STAGES = {
    1: [".github/workflows/dvc_stage1.yml"],
    2: [".github/workflows/stage2_smoke.yml", "pipeline/prepare_stage2.py", "kaggle/stage2_smoke.py"],
    3: [".github/workflows/stage3_finetune.yml", "kaggle/stage3_finetune.py"],
    4: [".github/workflows/stage4_two_models.yml", "kaggle/stage4_two_models.py"],
    5: [".github/workflows/stage5_mlflow_gate.yml", "pipeline/stage5_evaluate.py"],
    6: [".github/workflows/stage6_independent_eval.yml", "pipeline/stage6_evaluate.py"],
    7: [".github/workflows/stage7_continuous_training.yml", "pipeline/stage7_prepare.py",
        "pipeline/stage7_gate.py", "pipeline/stage7_promote.py", "kaggle/stage7_train.py"],
}
CHECKPOINTS = {
    "mobilenetv3": "mobilenet_model/mobilenet_v3_small_best.pth",
    "resnet18": "resnet_model/resnet18_best.pth",
    "custom_cnn": "cnn_model/optimized_custom_cnn_best.pth",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def readable_size(size):
    if size is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024


def safe_url(url):
    """Prevent embedded usernames, tokens, query parameters leaking in reports."""
    url = url.strip()
    try:
        x = urlsplit(url)
        if x.scheme and x.netloc:
            host = x.hostname or ""
            if x.port:
                host += f":{x.port}"
            return urlunsplit((x.scheme, host, x.path, "", ""))
    except ValueError:
        pass
    # Redact credentials from scp-style git URLs as well.
    return re.sub(r"[^\s/@]+@", "[user]@", url) if "@" in url else url


def git(root, *args):
    try:
        p = subprocess.run(["git", *args], cwd=root, text=True, encoding="utf-8",
                           errors="replace", capture_output=True, timeout=15, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def file_info(path):
    return {"exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None}


def count_images(path):
    if not path.is_dir():
        return None
    n = 0
    for base, _, files in os.walk(path):
        n += sum(Path(name).suffix.lower() in IMAGE_SUFFIXES for name in files)
    return n


def json_load(path):
    if not path.is_file():
        return None, "not found"
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f), None
    except (ValueError, OSError, UnicodeError) as e:
        return None, f"invalid JSON: {type(e).__name__}"


def find_json(root, base, depth_limit=4):
    """Small, bounded scan to avoid walking all PKLot images."""
    if not base.is_dir():
        return []
    found = []
    start_depth = len(base.parts)
    for folder, dirs, files in os.walk(base):
        current = Path(folder)
        if len(current.parts) - start_depth >= depth_limit:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in {".git", ".dvc", ".venv", "venv", "__pycache__"}]
        found.extend(str((current / n).relative_to(root)).replace("\\", "/")
                     for n in files if n.lower().endswith(".json") and len(found) < 100)
        if len(found) >= 100:
            break
    return found


def inspect_online(root):
    pred = root / "data/realtime_stream/predictions"
    saved = root / "data/confirmed_online/frames"
    labels = sorted(saved.glob("*/labels.json")) if saved.is_dir() else []
    images = sum(1 for _ in saved.glob("*/*.png")) if saved.is_dir() else 0
    pred_n = sum(1 for _ in pred.glob("*.json")) if pred.is_dir() else 0
    label_counts = {"Empty": 0, "Occupied": 0, "unknown": 0}
    invalid = 0
    sources = set()
    for p in labels:
        obj, err = json_load(p)
        if err or not isinstance(obj, dict):
            invalid += 1
            continue
        if isinstance(obj.get("source_image"), str):
            sources.add(obj["source_image"])
        slots = obj.get("slots")
        if not isinstance(slots, list):
            invalid += 1
            continue
        for slot in slots:
            lbl = slot.get("confirmed_label") if isinstance(slot, dict) else None
            label_counts[lbl if lbl in label_counts else "unknown"] += 1
    manifest, manifest_error = json_load(root / "data/realtime_stream/manifest.json")
    stream = {
        "manifest_exists": manifest_error is None,
        "manifest_frames": len(manifest.get("images", [])) if isinstance(manifest, dict) and isinstance(manifest.get("images"), list) else None,
        "prediction_files": pred_n,
        "confirmed_frames": len(labels),
        "confirmed_crops": images,
        "confirmed_label_counts": label_counts,
        "invalid_label_files": invalid,
        "distinct_source_images": len(sources),
        "source_note": "Prior project context: simulated online frames originate from PKLot test; do not train on these and evaluate on unchanged PKLot test.",
    }
    approved = root / "approved_online"
    approval, approval_error = json_load(approved / "manifest.json")
    stream["approved_online"] = {
        "directory_exists": approved.is_dir(),
        "dvc_pointer_exists": (root / "approved_online.dvc").is_file(),
        "manifest_exists": approval_error is None,
        "image_count": count_images(approved / "images"),
        "manifest_top_level_keys": sorted(approval.keys()) if isinstance(approval, dict) else [],
    }
    # Never assert the provenance of approved_online from the filename alone.
    return stream


def inspect_workflow(path):
    if not path.is_file():
        return {"exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    # Only summarize event types and references; do not include raw workflow or secrets.
    return {
        "exists": True,
        "manual_dispatch_mentioned": "workflow_dispatch" in text,
        "push_event_mentioned": bool(re.search(r"(?m)^\s*push\s*:", text)),
        "approved_online_pointer_mentioned": "approved_online.dvc" in text,
        "uses_kaggle": "kaggle" in text.lower(),
        "uses_dvc": "dvc" in text.lower(),
        "uses_mlflow": "mlflow" in text.lower(),
        "upload_artifact_mentioned": "upload-artifact" in text,
    }


def inspect_artifacts(root):
    # Inspect filenames inside ZIP without extraction, deserialization or executing code.
    locations = [root, root / "artifacts", root / "downloads", root / "outputs"]
    zips = []
    for base in locations:
        if not base.is_dir():
            continue
        for p in sorted(base.glob("*.zip"))[:40]:
            if "stage" not in p.name.lower() and "parking" not in p.name.lower():
                continue
            item = {"file": str(p.relative_to(root)).replace("\\", "/"), "size": readable_size(p.stat().st_size)}
            try:
                with zipfile.ZipFile(p) as z:
                    members = z.namelist()
                    item["file_count"] = len(members)
                    item["report_files"] = [m for m in members if m.endswith("_report.json")][:20]
                    item["checkpoint_files"] = [m for m in members if m.endswith(".pth")][:20]
            except (OSError, zipfile.BadZipFile):
                item["zip_error"] = "cannot inspect"
            zips.append(item)
    return zips


def collect(root):
    r = {
        "audit_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root": str(root),
        "audit_mode": "read-only, offline, no credentials collected, no inference/training executed",
    }
    branch = git(root, "branch", "--show-current")
    remotes = git(root, "remote", "-v")
    remote_rows = []
    if remotes:
        for line in remotes.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                remote_rows.append({"name": parts[0], "url_sanitized": safe_url(parts[1])})
    status = git(root, "status", "--porcelain")
    r["git"] = {
        "repository_detected": git(root, "rev-parse", "--is-inside-work-tree") == "true",
        "branch": branch,
        "commit": git(root, "rev-parse", "--short", "HEAD"),
        "last_commit_message": git(root, "log", "-1", "--format=%s"),
        "working_tree_changed_count": len(status.splitlines()) if status is not None and status else 0 if status == "" else None,
        "remote_names_and_sanitized_urls": remote_rows,
        "note": "Offline audit cannot confirm the latest commit is pushed or GitHub Actions completed.",
    }
    r["stages"] = {}
    for num, paths in STAGES.items():
        present = [p for p in paths if (root / p).is_file()]
        r["stages"][str(num)] = {
            "source_files_present": present,
            "missing_source_files": [p for p in paths if p not in present],
            "code_present": len(present) == len(paths),
            "run_verified_by_this_audit": False,
        }
    r["workflows"] = {
        p: inspect_workflow(root / p)
        for files in STAGES.values() for p in files if p.endswith((".yml", ".yaml"))
    }
    r["dvc"] = {
        "dvc_config_exists": (root / ".dvc/config").is_file(),
        "data_pointer": file_info(root / "data.dvc"),
        "approved_online_pointer": file_info(root / "approved_online.dvc"),
        "model_pointers": {k: file_info(root / (v + ".dvc")) for k, v in CHECKPOINTS.items()},
        "note": "Does not run dvc pull/push; remote availability is not verified.",
    }
    r["models"] = {k: file_info(root / v) for k, v in CHECKPOINTS.items()}
    data = root / "data"
    r["dataset"] = {
        "data_directory_exists": data.is_dir(),
        "train_image_files": count_images(data / "train"),
        "valid_image_files": count_images(data / "valid"),
        "val_image_files": count_images(data / "val"),
        "test_image_files": count_images(data / "test"),
        "possible_annotation_json": [x for x in find_json(root, data, depth_limit=2)
                                     if "annotations" in x.lower() or "coco" in x.lower()][:30],
    }
    r["online"] = inspect_online(root)
    r["local_artifact_zips"] = inspect_artifacts(root)
    r["mlflow"] = {
        "mlflow_tracking_uri_env_present": bool(os.environ.get("MLFLOW_TRACKING_URI")),
        "dagshub_token_env_present": bool(os.environ.get("DAGSHUB_TOKEN")),
        "kaggle_token_env_present": bool(os.environ.get("KAGGLE_API_TOKEN")),
        "remote_tracking_verified": False,
        "note": "Credential values and MLflow server are deliberately not accessed.",
    }
    findings = []
    missing_stage = [n for n, s in r["stages"].items() if not s["code_present"]]
    if missing_stage:
        findings.append("Missing expected stage code files for stages: " + ", ".join(missing_stage) + ". Names may differ; check manually.")
    if not r["online"]["approved_online"]["dvc_pointer_exists"]:
        findings.append("No approved_online.dvc pointer found: Stage 7 auto-trigger data batch not yet evidenced locally.")
    if r["online"]["confirmed_frames"]:
        findings.append("Confirmed online simulation data exists; do not reuse PKLot test-derived crops for training evaluated on PKLot test.")
    if not r["local_artifact_zips"]:
        findings.append("No Stage artifact ZIP found in top-level/outputs/artifacts/downloads folders; GitHub artifacts may exist remotely.")
    findings.append("Source-code presence does not prove Stage 7 end-to-end success; verify GitHub Actions/Kaggle/MLflow and approval/rollback separately.")
    r["findings"] = findings
    return r


def to_text(r):
    lines = ["SMART PARKING MLOPS — PROJECT AUDIT", "=" * 50,
             f"UTC: {r['audit_utc']}", f"Root: {r['project_root']}",
             f"Git: branch={r['git']['branch']} commit={r['git']['commit']} changed={r['git']['working_tree_changed_count']}",
             "", "STAGE FILES (presence != successful execution)"]
    for n, item in r["stages"].items():
        lines.append(f"Stage {n}: {'FILES_OK' if item['code_present'] else 'MISSING_FILES'}")
        for p in item["missing_source_files"]:
            lines.append(f"  missing: {p}")
    lines += ["", "CHECKPOINTS"]
    for n, info in r["models"].items():
        pointer = r["dvc"]["model_pointers"][n]
        lines.append(f"{n}: pth={info['exists']} ({readable_size(info['size_bytes'])}), dvc_pointer={pointer['exists']}")
    ds = r["dataset"]
    lines += ["", "DATASET", f"data/: {ds['data_directory_exists']}",
              f"train images: {ds['train_image_files']}, valid: {ds['valid_image_files']}, val: {ds['val_image_files']}, test: {ds['test_image_files']}"]
    o = r["online"]
    lines += ["", "ONLINE SIMULATION", f"manifest frames: {o['manifest_frames']}",
              f"predictions: {o['prediction_files']}", f"confirmed frames: {o['confirmed_frames']}",
              f"confirmed crops: {o['confirmed_crops']}",
              f"labels: {o['confirmed_label_counts']}",
              f"approved_online exists: {o['approved_online']['directory_exists']}",
              f"approved_online.dvc exists: {o['approved_online']['dvc_pointer_exists']}"]
    lines += ["", "LOCAL ARTIFACT ZIP FILES"]
    for z in r["local_artifact_zips"]:
        lines.append(f"- {z['file']} ({z['size']}) reports={z.get('report_files', [])}")
    if not r["local_artifact_zips"]:
        lines.append("(none found in searched locations)")
    lines += ["", "FINDINGS"] + ["- " + s for s in r["findings"]]
    lines += ["", "NOTE: This audit does not access secrets, trigger jobs, modify project data or verify remote runs."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Read-only Smart Parking MLOps project audit")
    parser.add_argument("--root", default=".", help="Project directory (default current directory)")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        parser.error("project folder does not exist")
    report = collect(root)
    out_dir = root / "project_audit"
    out_dir.mkdir(exist_ok=True)
    json_file = out_dir / "smart_parking_audit.json"
    txt_file = out_dir / "smart_parking_audit.txt"
    json_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_file.write_text(to_text(report), encoding="utf-8")
    print(to_text(report))
    print("Report files:", txt_file, "and", json_file)
    print("Only the project_audit/ reports were written; no training or cloud action was performed.")


if __name__ == "__main__":
    main()
