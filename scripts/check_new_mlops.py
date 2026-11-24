#!/usr/bin/env python3
"""Read-only preflight for GitHub Actions -> DVC -> Kaggle Private Dataset -> Kaggle GPU.
Run from the Smart Parking project root. Never uploads, creates or changes remote resources.
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
DATASET_SLUG = "nguynngthibo/smart-parking-mlops-training"
KERNEL_SLUG = "nguynngthibo/smart-parking-retraining"
CHECKS = {}

def add(name, status, detail=""):
    CHECKS[name] = {"status": status, "detail": detail}
    print(f"[{status:5}] {name}: {detail}")

def run(cmd, timeout=35, cwd=ROOT):
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           stdin=subprocess.DEVNULL)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "command not installed"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except Exception as exc:
        return 125, "", type(exc).__name__

def first_line(s):
    return (s.strip().splitlines() or [""])[0][:180]

def exists_file(name, path):
    ok = path.is_file()
    add(name, "PASS" if ok else "FAIL", str(path.relative_to(ROOT)))
    return ok

def main():
    print("SMART PARKING - NEW MLOPS PIPELINE PRECHECK (READ ONLY)")
    print("Root:", ROOT)
    print("No credentials will be displayed or saved.\n")

    for name, rel in [
        ("file.data_dvc", "data.dvc"),
        ("file.dvc_config", ".dvc/config"),
        ("file.retrain", "retraining/retrain.py"),
        ("file.kaggle_train", "kaggle/train.py"),
        ("file.kernel_metadata", "kaggle/kernel-metadata.json"),
    ]:
        exists_file(name, ROOT / rel)

    # Kaggle notebook configuration (safe metadata only)
    mp = ROOT / "kaggle/kernel-metadata.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8-sig"))
            src = meta.get("dataset_sources", [])
            good = meta.get("id") == KERNEL_SLUG and meta.get("enable_gpu") is True
            add("kaggle.metadata", "PASS" if good else "WARN",
                f"id_ok={meta.get('id') == KERNEL_SLUG}; gpu={meta.get('enable_gpu')}; "
                f"dataset_attached={DATASET_SLUG in src}")
        except (OSError, ValueError) as exc:
            add("kaggle.metadata", "FAIL", type(exc).__name__)

    # Project dataset readiness: only counts files; no labels read/transferred
    base = ROOT / "data"
    if base.exists():
        for split in ("train", "valid", "test"):
            d = base / split
            add(f"data.{split}", "PASS" if d.is_dir() else "WARN",
                "directory present" if d.is_dir() else "not present locally (DVC pull may be required)")
        confirmed = base / "confirmed_online"
        add("data.confirmed_online", "PASS" if confirmed.is_dir() else "WARN",
            "directory present" if confirmed.is_dir() else "not present locally")
    else:
        add("data.local", "WARN", "data/ absent; GitHub Actions will need DVC pull")

    # GitHub Actions secrets cannot be read with this local checker
    workflows = ROOT / ".github/workflows"
    add("github.workflow", "PASS" if workflows.exists() and list(workflows.glob('*.y*ml')) else "WARN",
        "workflow file(s) found" if workflows.exists() and list(workflows.glob('*.y*ml')) else "no GitHub Actions workflow found yet")
    rc, out, err = run(["git", "rev-parse", "--is-inside-work-tree"])
    add("git.repository", "PASS" if rc == 0 and out.strip() == "true" else "FAIL")
    rc, out, err = run(["git", "remote", "get-url", "origin"])
    # Only report host, not URL (a remote URL could contain credentials).
    add("git.origin", "PASS" if rc == 0 else "WARN",
        "configured" if rc == 0 else "origin not configured")

    # DVC config and read-only remote connectivity check
    if shutil.which("dvc"):
        rc, out, err = run(["dvc", "--version"])
        add("dvc.installed", "PASS" if rc == 0 else "FAIL", first_line(out))
        rc, out, err = run(["dvc", "remote", "list"], timeout=20)
        # Redact all remote URLs: user-configured URLs could contain passwords.
        add("dvc.remote_config", "PASS" if rc == 0 and out.strip() else "WARN",
            "remote configured" if rc == 0 and out.strip() else "no remote or command failed")
        # No dvc pull/status -c by default: may be slow or require credentials.
        if os.getenv("CHECK_DVC_REMOTE") == "1":
            rc, out, err = run(["dvc", "status", "-c"], timeout=180)
            add("dvc.remote_read", "PASS" if rc == 0 else "FAIL",
                f"dvc status -c exit={rc}; this does not prove full download permission")
        else:
            add("dvc.remote_read", "SKIP", "set CHECK_DVC_REMOTE=1 to test dvc status -c")
    else:
        add("dvc.installed", "FAIL", "dvc executable not found")

    # Check local Python dependencies without installing anything.
    for mod in ("kaggle", "torch", "mlflow"):
        try:
            __import__(mod)
            add(f"python.{mod}", "PASS", "importable")
        except Exception as exc:
            add(f"python.{mod}", "WARN", type(exc).__name__)

    # Kaggle authentication: list owned datasets, no secret required on Kaggle GPU.
    if shutil.which("kaggle"):
        rc, out, err = run(["kaggle", "--version"])
        add("kaggle.cli", "PASS" if rc == 0 else "WARN", first_line(out or err))
        rc, out, err = run(["kaggle", "datasets", "list", "--mine", "--page", "1"], timeout=60)
        add("kaggle.api_auth", "PASS" if rc == 0 else "FAIL",
            "authenticated listing succeeded" if rc == 0 else f"listing failed (exit={rc}); check Kaggle API credentials/network")
        rc, out, err = run(["kaggle", "datasets", "create", "--help"], timeout=15)
        add("kaggle.dataset_create_command", "PASS" if rc == 0 else "FAIL",
            "available (not executed)" if rc == 0 else "not available")
        rc, out, err = run(["kaggle", "datasets", "version", "--help"], timeout=15)
        add("kaggle.dataset_version_command", "PASS" if rc == 0 else "FAIL",
            "available (not executed)" if rc == 0 else "not available")
        rc, out, err = run(["kaggle", "kernels", "push", "--help"], timeout=15)
        add("kaggle.kernel_push_command", "PASS" if rc == 0 else "FAIL",
            "available (not executed)" if rc == 0 else "not available")
        if CHECKS["kaggle.api_auth"]["status"] == "PASS":
            rc, out, err = run(["kaggle", "datasets", "status", DATASET_SLUG], timeout=45)
            # A missing dataset is expected before creation.
            add("kaggle.private_dataset_exists", "PASS" if rc == 0 else "WARN",
                "dataset exists" if rc == 0 else "not accessible or not created yet")
    else:
        add("kaggle.cli", "FAIL", "kaggle executable not found")

    add("github.actions_secrets", "MANUAL", "verify DAGSHUB_TOKEN and Kaggle API credentials in repository Actions Secrets")
    add("dataset.license", "MANUAL", "verify PKLot and online-data redistribution terms before upload")
    add("data.leakage", "WARN", "confirmed_online crops sourced from PKLot test: separate them from unbiased final evaluation")
    add("gpu.training", "SKIP", "actual Kaggle GPU retraining requires a later end-to-end test")

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "project_root": str(ROOT), "dataset_slug": DATASET_SLUG,
              "kernel_slug": KERNEL_SLUG, "read_only": True, "checks": CHECKS}
    dst = ROOT / "outputs" / "new_mlops_preflight.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {dst}")
    print("PASS does not mean full automation works; no dataset upload or GPU job was launched.")

if __name__ == "__main__":
    main()
