
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MODELS = {
    "mobilenetv3": {
        "checkpoint": "mobilenet_model/mobilenet_v3_small_best.pth",
        "architecture": "mobilenet_v3_small",
    },
    "resnet18": {
        "checkpoint": "resnet_model/resnet18_best.pth",
        "architecture": "resnet18",
    },
    "custom_cnn": {
        "checkpoint": "cnn_model/optimized_custom_cnn_best.pth",
        "architecture": "OptimizedPKLotCNN",
    },
}

REQUIRED_CODE = [
    "api/main.py",
    "dashboard/app.py",
    "dashboard/online_stream.py",
    "prepare_online.py",
    "validate_online_data.py",
    "retraining/evaluate.py",
    "retraining/retrain.py",
]

COUNTS = Counter()
ISSUES = []
FINDINGS = []


def section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def check(ok, message, important=False):
    status = "PASS" if ok else "MISSING"
    print(f"[{status}] {message}")

    COUNTS[status] += 1

    if not ok and important:
        ISSUES.append(message)


def note(message):
    print("[INFO]", message)
    FINDINGS.append(message)


def warning(message):
    print("[WARN]", message)
    ISSUES.append(message)
    COUNTS["WARN"] += 1


def read_text(path):
    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace"
        )
    except OSError:
        return ""


def read_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        warning(
            f"Cannot read JSON: {path.relative_to(ROOT)} "
            f"({type(e).__name__})"
        )
        return None


def run_local(command):
    """
    Read-only local commands.
    Does not invoke git push, DVC push or cloud APIs.
    """
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        p = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=25,
            check=False,
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired):
        return -1, "", "command unavailable"


def code_files():
    """
    Scan project source, excluding datasets,
    model storage, Git, DVC cache and environments.
    """
    skip_dirs = {
        ".git", ".dvc", ".venv", "venv",
        "__pycache__", ".pytest_cache",
        "data", "mlruns", "outputs",
        ".mypy_cache",
    }

    result = []

    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [
            d for d in dirs
            if d not in skip_dirs
            and not d.startswith("mlruns")
        ]

        for name in files:
            if name.endswith(".py"):
                result.append(Path(folder) / name)

    return sorted(result)


def inspect_python():
    section("1. PYTHON SOURCE CODE")

    scripts = code_files()

    print("Python files:", len(scripts))

    failed = 0

    for file in scripts:
        relative = file.relative_to(ROOT)

        try:
            content = read_text(file)
            tree = ast.parse(content, filename=str(file))

            functions = [
                n.name for n in ast.walk(tree)
                if isinstance(
                    n,
                    (ast.FunctionDef, ast.AsyncFunctionDef)
                )
            ]

            classes = [
                n.name for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef)
            ]

            print(
                f"[OK] {relative} | "
                f"{len(content.splitlines())} lines | "
                f"{len(functions)} functions | "
                f"{len(classes)} classes"
            )

            if str(relative).replace("\\", "/") in {
                "retraining/retrain.py",
                "retraining/evaluate.py",
                "api/main.py",
            }:
                print("     Functions:", ", ".join(functions[:15]))
                print("     Classes:", ", ".join(classes[:12]))

        except SyntaxError as e:
            failed += 1
            warning(
                f"Python syntax error: {relative}:"
                f"{e.lineno}: {e.msg}"
            )

    check(failed == 0, "Python syntax validation")

    print("\nRequired application files:")

    for relative in REQUIRED_CODE:
        check(
            (ROOT / relative).is_file(),
            relative,
            important=True,
        )


def inspect_models():
    section("2. THREE MODEL CHECKPOINTS")

    for name, model_info in MODELS.items():
        path = ROOT / model_info["checkpoint"]

        print(f"\nModel: {name}")
        print("Architecture:", model_info["architecture"])
        print("Path:", model_info["checkpoint"])

        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"[PASS] Checkpoint: {size_mb:.2f} MB")
            COUNTS["PASS"] += 1
        else:
            check(
                False,
                f"{name}: checkpoint missing",
                important=True,
            )

    api = read_text(ROOT / "api/main.py")

    section("3. FASTAPI MODEL SUPPORT")

    for name in MODELS:
        check(
            name in api,
            f"FastAPI references {name}"
        )

    check(
        "get_model_version_by_alias" in api,
        "MLflow Registry alias lookup"
    )

    check(
        "load_registry_model" in api,
        "Registry model loading"
    )

    check(
        "load_local_model" in api,
        "Local model fallback"
    )

    print("\nRegistry review:")

    for name in MODELS:
        if name == "mobilenetv3":
            note(
                "MobileNetV3: registry handling exists in "
                "previously reviewed API code"
            )
        else:
            note(
                f"{name}: registry version and alias "
                "must be verified separately"
            )


def inspect_coco(split):
    directory = DATA / split
    path = directory / "_annotations.coco.json"

    if not path.is_file():
        check(False, f"{split}: COCO JSON missing")
        return

    obj = read_json(path)

    if not isinstance(obj, dict):
        return

    images = obj.get("images", [])
    annotations = obj.get("annotations", [])
    categories = obj.get("categories", [])

    if not isinstance(images, list):
        warning(f"{split}: images is not a list")
        return

    if not isinstance(annotations, list):
        warning(f"{split}: annotations is not a list")
        return

    category_names = {
        item.get("id"): item.get("name")
        for item in categories
        if isinstance(item, dict)
    }

    category_counts = Counter()
    image_ids = set()

    for image in images:
        if isinstance(image, dict):
            image_ids.add(image.get("id"))

    orphans = 0

    for ann in annotations:
        if not isinstance(ann, dict):
            continue

        category_counts[ann.get("category_id")] += 1

        if ann.get("image_id") not in image_ids:
            orphans += 1

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Annotations:", len(annotations))
    print("Categories:", category_names)
    print("Counts:", dict(category_counts))
    print("Orphan annotations:", orphans)

    check(orphans == 0, f"{split}: COCO references valid")

    # Check names without opening all images.
    missing = 0

    for image in images:
        filename = image.get("file_name", "")
        if not (directory / filename).is_file():
            missing += 1

    check(
        missing == 0,
        f"{split}: {missing} missing image files"
    )


def inspect_data():
    section("4. PKLOT DATASET")

    for split in ("train", "valid", "test"):
        inspect_coco(split)


def inspect_online():
    section("5. ONLINE SIMULATION / CONFIRMED DATA")

    online = DATA / "realtime_stream"
    manifest_path = online / "manifest.json"
    coco_path = online / "_annotations.coco.json"

    check(manifest_path.is_file(), "Online manifest")
    check(coco_path.is_file(), "Online COCO JSON")

    manifest = (
        read_json(manifest_path)
        if manifest_path.is_file()
        else None
    )

    if isinstance(manifest, dict):
        entries = manifest.get("images", [])
        print("Online manifest frames:", len(entries))

        missing = 0

        for item in entries:
            filename = Path(item.get("file_name", "")).name

            if not (online / "images" / filename).is_file():
                missing += 1

        check(
            len(entries) == 20 and missing == 0,
            f"20 online images; missing={missing}"
        )

    online_coco = (
        read_json(coco_path)
        if coco_path.is_file()
        else None
    )

    if isinstance(online_coco, dict):
        print(
            "Online COCO annotations:",
            len(online_coco.get("annotations", []))
        )

    confirmed_dir = DATA / "confirmed_online" / "frames"
    label_files = sorted(
        confirmed_dir.glob("*/labels.json")
    )

    print("\nConfirmed frames:", len(label_files))

    distribution = Counter()
    corrections = 0
    total_slots = 0
    missing_crops = 0
    invalid_labels = 0
    source_names = set()
    model_versions = Counter()

    for label_path in label_files:
        data = read_json(label_path)

        if not isinstance(data, dict):
            continue

        source_names.add(data.get("source_image"))
        model_versions[str(data.get("model_version"))] += 1

        for slot in data.get("slots", []):
            total_slots += 1

            label = slot.get("confirmed_label")
            predicted = slot.get("predicted_label")
            label_id = slot.get("label_id")

            distribution[label] += 1

            expected = {
                "Empty": 0,
                "Occupied": 1
            }.get(label)

            if expected is None or label_id != expected:
                invalid_labels += 1

            if predicted != label:
                corrections += 1

            filename = slot.get("crop_file")

            if not isinstance(filename, str):
                missing_crops += 1
                continue

            # Prevent paths outside the frame directory.
            if Path(filename).name != filename:
                invalid_labels += 1
                continue

            if not (label_path.parent / filename).is_file():
                missing_crops += 1

    print("Total crop labels:", total_slots)
    print("Distribution:", dict(distribution))
    print("Human corrections:", corrections)
    print("Unique source images:", len(source_names))
    print("Source model versions:", dict(model_versions))
    print("Missing crops:", missing_crops)
    print("Invalid labels:", invalid_labels)

    check(len(label_files) == 20, "20 confirmed frames")
    check(total_slots == 2000, "2000 crop labels")

    check(
        missing_crops == 0 and invalid_labels == 0,
        "Confirmed data structure",
        important=True,
    )

    if corrections == 0 and total_slots:
        warning(
            "All confirmed labels equal AI predictions; "
            "human review quality not independently verified"
        )

    if source_names:
        warning(
            "Online images originated from PKLot Test. "
            "Exclude these images from v2 test evaluation."
        )


def inspect_retraining():
    section("6. RETRAINING SUPPORT FOR THREE MODELS")

    train_path = ROOT / "retraining/retrain.py"
    eval_path = ROOT / "retraining/evaluate.py"

    train_code = read_text(train_path)
    eval_code = read_text(eval_path)

    check(train_path.is_file(), "Retraining script exists")
    check(eval_path.is_file(), "Evaluation script exists")

    check(
        "ConfirmedOnlineDataset" in train_code,
        "Retraining loads confirmed online crops"
    )

    check(
        "CocoCropDataset" in train_code,
        "Retraining includes old PKLot replay data"
    )

    check(
        "load_model" in train_code,
        "Retraining loads existing checkpoint"
    )

    for name in MODELS:
        train_ref = name in train_code
        eval_ref = name in eval_code

        check(
            train_ref,
            f"Retrain script explicitly references {name}"
        )

        check(
            eval_ref,
            f"Evaluate script explicitly references {name}"
        )

    check(
        "--model" in train_code,
        "CLI supports selecting architecture using --model"
    )

    check(
        "evaluation_gate" in train_code,
        "Evaluation Gate present"
    )

    check(
        "mlflow" in train_code,
        "MLflow tracking integration present"
    )

    if "production" in train_code.lower():
        warning(
            "Review retraining script for automatic "
            "Production promotion before deployment"
        )

    note(
        "Static checks indicate code support only; "
        "they do not prove checkpoint compatibility "
        "or successful fine-tuning."
    )


def inspect_git_dvc():
    section("7. GIT / GITHUB / DAGSHUB / DVC")

    rc, head, _ = run_local(
        ["git", "log", "-1", "--oneline"]
    )

    if rc == 0:
        print("HEAD:", head)
    else:
        warning("Unable to read Git HEAD")

    rc, status, _ = run_local([
        "git", "status", "--short",
        "--untracked-files=normal"
    ])

    if rc == 0:
        changes = status.splitlines() if status else []
        print("Working tree changes:", len(changes))

        for line in changes[:30]:
            print(" ", line)
    else:
        warning("Unable to inspect Git working tree")

    rc, remotes, _ = run_local(["git", "remote"])

    if rc == 0:
        print("Git remote names:", remotes.splitlines())
    else:
        warning("Git remotes unavailable")

    rc, branches, _ = run_local([
        "git", "branch", "-vv"
    ])

    if rc == 0:
        print("Branches:", branches)

    rc, tracked, _ = run_local(["git", "ls-files"])

    if rc == 0:
        tracked_set = set(tracked.splitlines())

        for relative in REQUIRED_CODE:
            check(
                relative in tracked_set,
                f"Git tracks {relative}"
            )

        for name in (
            "data.dvc",
            ".dvc/config",
            ".gitignore"
        ):
            check(name in tracked_set, f"Git tracks {name}")

        suspicious = [
            name for name in tracked_set
            if (
                name.endswith((".pth", ".pt", ".db"))
                or name.startswith("data/")
                or name in (".env", "kaggle.json")
            )
        ]

        check(
            not suspicious,
            f"No dataset/checkpoint/credentials "
            f"tracked directly: {suspicious}"
        )

    dvc_metadata = ROOT / "data.dvc"
    dvc_config = ROOT / ".dvc/config"

    check(dvc_metadata.is_file(), "data.dvc exists")
    check(dvc_config.is_file(), ".dvc/config exists")

    if dvc_metadata.is_file():
        content = read_text(dvc_metadata)

        print("DVC metadata bytes:", dvc_metadata.stat().st_size)

        check(
            "path: data" in content or
            "path: 'data'" in content or
            'path: "data"' in content,
            "data.dvc tracks data directory"
        )

    rc, output, _ = run_local([
        "dvc", "remote", "list"
    ])

    if rc == 0:
        # Do not print credentials or private remote URLs.
        names = [
            line.split()[0]
            for line in output.splitlines()
            if line.strip()
        ]
        print("DVC remote names:", names)
        check(bool(names), "DVC remote configured")
    else:
        warning("DVC CLI unavailable or remote check failed")

    note(
        "Remote DVC sync, MLflow Registry access and "
        "GitHub push status are not verified online."
    )


def inspect_mlflow():
    section("8. MLFLOW / MODEL REGISTRY")

    db_files = []

    for model_dir in (
        "mobilenet_model",
        "resnet_model",
        "cnn_model",
    ):
        path = ROOT / model_dir

        if not path.exists():
            continue

        db_files.extend(path.glob("*.db"))

        mlruns = path / "mlruns"

        print(
            model_dir,
            "| MLflow DB:",
            len(list(path.glob("*.db"))),
            "| mlruns:",
            mlruns.is_dir()
        )

    print("Total local MLflow DB:", len(db_files))

    source_files = code_files()

    for name, info in MODELS.items():
        matches = []

        for file in source_files:
            content = read_text(file)

            if (
                "mlflow" in content.lower()
                and (
                    name in content.lower()
                    or info["architecture"].lower()
                    in content.lower()
                )
            ):
                matches.append(str(file.relative_to(ROOT)))

        print(f"{name}: MLflow-related files = {matches}")

    note(
        "Local MLflow files do not prove that all "
        "three models have registered versions on DagsHub."
    )


def inspect_cloud():
    section("9. KAGGLE / GITHUB ACTIONS")

    kaggle_dir = ROOT / "kaggle"
    workflow_dir = ROOT / ".github/workflows"

    check(
        kaggle_dir.is_dir(),
        "Kaggle configuration directory"
    )

    metadata = list(
        kaggle_dir.glob("*metadata.json")
    ) if kaggle_dir.exists() else []

    check(
        bool(metadata),
        "Kaggle kernel metadata"
    )

    workflows = (
        list(workflow_dir.glob("*.yml"))
        + list(workflow_dir.glob("*.yaml"))
        if workflow_dir.is_dir() else []
    )

    print("GitHub workflows:", [
        p.name for p in workflows
    ])

    retrain_workflows = []

    for path in workflows:
        content = read_text(path).lower()

        if (
            "kaggle" in content
            or "retrain" in path.name.lower()
        ):
            retrain_workflows.append(path.name)

    check(
        bool(retrain_workflows),
        f"Retraining workflows: {retrain_workflows}"
    )

    rc, stdout, _ = run_local([
        "kaggle", "--version"
    ])

    check(
        rc == 0,
        f"Kaggle CLI installed: {stdout if rc == 0 else 'No'}"
    )

    note(
        "Kaggle credentials, GPU quota and cloud "
        "execution are intentionally not tested."
    )


def summary():
    section("10. AUDIT SUMMARY")

    print("PASS:", COUNTS["PASS"])
    print("MISSING:", COUNTS["MISSING"])
    print("WARN:", COUNTS["WARN"])

    print("\nImportant findings:")

    if ISSUES:
        for item in ISSUES:
            print(" -", item)
    else:
        print(" No issues detected by these static checks.")

    print("\nNEXT DEVELOPMENT PRIORITIES")
    print("1. Verify Registry v1 for all three models.")
    print("2. Verify checkpoint loading for each model.")
    print("3. Extend retrain/evaluate for 3 architectures.")
    print("4. Test fine-tuning manually on Kaggle GPU.")
    print("5. Register candidate versions in MLflow.")
    print("6. Add GitHub Actions retraining workflow.")
    print("7. Implement per-model deployment and rollback.")

    print("\nAUDIT IS READ-ONLY.")
    print("No project files or models were modified.")


def main():
    print("=" * 65)
    print("SMART PARKING MLOPS - FULL PROJECT AUDIT")
    print("Project:", ROOT)
    print("Python:", sys.version.split()[0])
    print("Mode: READ ONLY / NO CLOUD OPERATIONS")
    print("=" * 65)

    inspect_python()
    inspect_models()
    inspect_data()
    inspect_online()
    inspect_retraining()
    inspect_git_dvc()
    inspect_mlflow()
    inspect_cloud()
    summary()


if __name__ == "__main__":
    main()
