#!/usr/bin/env python3
"""Read-only acceptance checks for Smart Parking MLOps demo.

No retraining, no MLflow writes, no DVC push, no Git changes.
With --inference, calls the existing FastAPI prediction endpoint twice on
one untouched, indexed PKLot test image; models may be loaded/cached in RAM.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def check(title, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {title}" + (f": {detail}" if detail else ""))
    return bool(ok)


def request_json(url, *, data=None, content_type=None, timeout=12):
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def multipart_payload(filename, raw, model_variant):
    boundary = "SmartParkingAccept" + uuid.uuid4().hex
    body = bytearray()
    for name, value in (("model_type", "mobilenetv3"), ("model_variant", model_variant)):
        body.extend((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode())
    body.extend((f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: image/jpeg\r\n\r\n").encode())
    body.extend(raw)
    body.extend((f"\r\n--{boundary}--\r\n").encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def find_test_image(root):
    split_dir = root / "data" / "test"
    annotations = split_dir / "_annotations.coco.json"
    if not annotations.is_file():
        raise FileNotFoundError("Không có data/test/_annotations.coco.json")
    doc = json.loads(annotations.read_text(encoding="utf-8"))
    supported = {int(row["image_id"]) for row in doc.get("annotations", []) if int(row.get("category_id", -1)) in (1, 2)}
    for row in doc.get("images", []):
        if int(row["id"]) in supported:
            path = split_dir / Path(str(row["file_name"])).name
            if path.is_file():
                return path
    raise FileNotFoundError("Không tìm thấy ảnh PKLot test có ô đỗ hợp lệ")


def run():
    parser = argparse.ArgumentParser(description="Read-only verification of Smart Parking demo")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--inference", action="store_true", help="Test an unchanged PKLot image with Production and QA (read-only)")
    args = parser.parse_args()
    root = args.root.resolve()
    api = args.api.rstrip("/")
    checks = []

    required = ["api/main.py", "api/qa_preview.py", "dashboard/app.py", "dashboard/model_review_ui.py", "dashboard/qa_preview_ui.py", "pipeline/stage7_batch5_worker.py"]
    missing = [f for f in required if not (root / f).is_file()]
    checks.append(check("Mã nguồn quan trọng", not missing, ", ".join(missing) if missing else "Đầy đủ"))

    try:
        cmd = subprocess.run(["dvc", "status"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90, check=False)
        clean = cmd.returncode == 0 and "Data and pipelines are up to date" in cmd.stdout
        checks.append(check("DVC", clean, (cmd.stdout or cmd.stderr).strip()[:200]))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        checks.append(check("DVC", False, "Không kiểm tra được; cần chạy trong môi trường có DVC"))

    try:
        cmd = subprocess.run(["git", "status", "--short"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
        if cmd.returncode:
            checks.append(check("Git status", False, cmd.stderr[:160]))
        else:
            tracked_changes = [line for line in cmd.stdout.splitlines() if not line.startswith("??")]
            if tracked_changes:
                print(f"[INFO] Git có {len(tracked_changes)} file thay đổi chưa commit; hãy kiểm tra và commit sau khi thử giao diện")
            else:
                print("[PASS] Không có thay đổi đối với file Git đã theo dõi")
            untracked = [line for line in cmd.stdout.splitlines() if line.startswith("??")]
            if untracked:
                print(f"[INFO] {len(untracked)} file/nhóm chưa theo dõi, hãy xem xét riêng (không tự xóa)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        checks.append(check("Git status", False, "Không thể gọi git"))

    health = None
    qa = None
    for title, endpoint in (("FastAPI health", "/health"), ("QA Registry metadata", "/qa-model-info")):
        try:
            info = request_json(api + endpoint)
            if endpoint == "/health":
                health = info
                valid = (str(info.get("status")) == "healthy" and
                         str(info.get("registry_alias")) == "production" and
                         str(info.get("model_source")) == "DagsHub Registry")
                checks.append(check(title, valid, f"version={info.get('model_version')}, source={info.get('model_source')}"))
            else:
                qa = info
                valid = (info.get("available") is True and info.get("review_status") == "QA_ONLY"
                         and info.get("test_only") is True and str(info.get("github_run_id")) == "45903240562")
                checks.append(check(title, valid, f"version={info.get('version')}, run={info.get('github_run_id')}"))
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            checks.append(check(title, False, f"{type(exc).__name__}: {str(exc)[:140]}"))
    if health and qa and qa.get("available") is True:
        checks.append(check("QA và Production tách biệt", str(health.get("model_version")) != str(qa.get("version")),
                            f"Production={health.get('model_version')}, QA={qa.get('version')}"))

    if args.inference:
        try:
            photo = find_test_image(root)
            data = photo.read_bytes()
            for variant, label in (("production", "Production"), ("qa_preview", "QA")):
                payload, content_type = multipart_payload(photo.name, data, variant)
                info = request_json(api + "/predict", data=payload, content_type=content_type, timeout=150)
                expected = (health or {}).get("model_version") if variant == "production" else (qa or {}).get("version")
                valid = (info.get("model_used") == "mobilenetv3" and expected is not None
                         and str(info.get("model_version")) == str(expected)
                         and bool(info.get("predictions")))
                checks.append(check(f"Suy luận {label}", valid, f"API version={info.get('model_version')}, expected={expected}"))
        except (OSError, urllib.error.URLError, ValueError, KeyError, TimeoutError) as exc:
            checks.append(check("Suy luận trên ảnh PKLot", False, f"{type(exc).__name__}: {str(exc)[:160]}"))

    print("\nKẾT LUẬN:", "PASS" if all(checks) else "CẦN KIỂM TRA", f"({sum(checks)}/{len(checks)} mục đạt)")
    print("Chỉ đọc dữ liệu; không cập nhật Registry, Production, Git hoặc DVC.")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(run())
