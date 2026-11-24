
import os
import ast
import json
import subprocess
import configparser
from pathlib import Path
from collections import Counter

# ============================================================
# SMART PARKING MLOPS - FULL PROJECT AUDIT
# READ-ONLY: KHONG CHINH SUA BAT KY FILE NAO
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".idea", ".vscode",
    "node_modules", "site-packages",
    "build", "dist", "cache", "tmp",
    "__results___files", "_results_files",
    ".mypy_cache"
}

# KhÃ´ng Ä‘i sÃ¢u vÃ o cÃ¡c thÆ° má»¥c dá»¯ liá»‡u lá»›n
SKIP_DEEP_SCAN = {
    ".dvc", "mlruns", "data", "datasets",
    "dataset", "runs", "outputs"
}

MODEL_EXTS = {".pth", ".pt", ".onnx"}

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tiff"
}

SECRET_NAMES = {
    ".env", "kaggle.json", "credentials.json",
    "token.json", "secrets.toml",
    "service-account.json", "id_rsa"
}

FEATURES = {
    "MLflow": ["mlflow"],
    "DagsHub": ["dagshub"],
    "DVC": ["dvc"],
    "Kaggle": ["kaggle"],
    "Streamlit": ["streamlit", "st."],
    "FastAPI": ["fastapi", "apirouter"],
    "ROI / Crop": [
        "crop", "bbox", "roi",
        "warpperspective",
        "getperspectivetransform"
    ],
    "Training": [
        "optimizer.step", "loss.backward",
        "scaler.step", ".backward("
    ],
    "Retraining": [
        "retrain", "fine_tune", "finetune"
    ],
    "Monitoring": [
        "data_drift", "drift_score",
        "monitoring"
    ],
    "Model Registry": [
        "register_model",
        "registered_model",
        "models:/",
        "set_registered_model_alias"
    ]
}


# ============================================================
# CAC HAM HO TRO
# ============================================================

def section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def rel(path):
    return str(path.relative_to(ROOT))


def is_secret(path):
    name = path.name.lower()

    return (
        name in SECRET_NAMES
        or name.startswith(".env.")
        or name.endswith(".pem")
        or name.endswith(".key")
    )


def size_mb(path):
    try:
        return round(
            path.stat().st_size / (1024 * 1024), 2
        )
    except OSError:
        return 0


def safe_walk(root):
    """
    Tráº£ vá» (folder, files).
    KhÃ´ng quÃ©t sÃ¢u cache vÃ  dá»¯ liá»‡u lá»›n.
    """
    for current, dirs, files in os.walk(root):
        folder = Path(current)

        dirs[:] = sorted(
            d for d in dirs
            if d not in IGNORE_DIRS
            and d not in SKIP_DEEP_SCAN
        )

        yield folder, sorted(files)


def collect_files():
    result = []

    for folder, files in safe_walk(ROOT):
        for name in files:
            path = folder / name

            if not is_secret(path):
                result.append(path)

    return result


def read_source(path):
    try:
        return path.read_text(
            encoding="utf-8-sig",
            errors="replace"
        )
    except OSError:
        return ""


ALL_FILES = collect_files()

PYTHON_FILES = sorted(
    p for p in ALL_FILES
    if p.suffix.lower() == ".py"
)

MODEL_FILES = sorted(
    p for p in ALL_FILES
    if p.suffix.lower() in MODEL_EXTS
)


# ============================================================
# 1. PROJECT OVERVIEW
# ============================================================

section("1. PROJECT OVERVIEW")

print("Project:", ROOT.name)
print("Root path:", ROOT)

print("Python scripts:", len(PYTHON_FILES))
print("Model files:", len(MODEL_FILES))

print("\nMain directories:")

for path in sorted(ROOT.iterdir()):
    if path.is_dir() and path.name not in IGNORE_DIRS:
        print(" -", path.name)


# ============================================================
# 2. SOURCE CODE ANALYSIS
# ============================================================

section("2. PYTHON SOURCE CODE / API / DASHBOARD")

for path in PYTHON_FILES:
    source = read_source(path)

    print(f"\n[FILE] {rel(path)}")
    print("  Lines:", len(source.splitlines()))

    if not source:
        print("  Status: Cannot read source")
        continue

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print("  Syntax error at line:", e.lineno)
        continue

    functions = []
    classes = []
    routes = []

    for node in ast.walk(tree):

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            functions.append(node.name)

            for decorator in node.decorator_list:

                if not isinstance(decorator, ast.Call):
                    continue

                func = decorator.func

                if not isinstance(func, ast.Attribute):
                    continue

                if func.attr not in {
                    "get", "post", "put",
                    "delete", "patch", "websocket"
                }:
                    continue

                route = "dynamic"

                if decorator.args:
                    arg = decorator.args[0]

                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                    ):
                        route = arg.value

                routes.append(
                    f"{func.attr.upper()} {route}"
                )

        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)

    print("  Functions:", len(functions))

    if functions:
        print(
            "  Function names:",
            ", ".join(functions[:30])
        )

    if classes:
        print(
            "  Classes:",
            ", ".join(classes[:20])
        )

    if routes:
        print(
            "  API routes:",
            ", ".join(routes)
        )

    detected = []
    lower_source = source.lower()

    for feature, keywords in FEATURES.items():
        if any(
            keyword.lower() in lower_source
            for keyword in keywords
        ):
            detected.append(feature)

    print(
        "  Detected features:",
        ", ".join(detected) if detected else "None"
    )


# ============================================================
# 3. DATASET / COCO
# ============================================================

section("3. DATASET / COCO")

DATA_DIR = ROOT / "data"

SPLITS = [
    "train", "valid", "test",
    "realtime_stream", "confirmed"
]


def count_images(folder):
    count = 0

    for current, dirs, files in os.walk(folder):
        dirs[:] = [
            d for d in dirs
            if d not in IGNORE_DIRS
        ]

        count += sum(
            Path(name).suffix.lower() in IMAGE_EXTS
            for name in files
        )

    return count


for split in SPLITS:
    folder = DATA_DIR / split

    print(f"\n[{split.upper()}]")

    if not folder.is_dir():
        print("  Status: NOT FOUND")
        continue

    print("  Images:", count_images(folder))

    coco_path = folder / "_annotations.coco.json"

    if not coco_path.is_file():
        print("  COCO: NOT FOUND")
        continue

    try:
        with coco_path.open(
            "r", encoding="utf-8"
        ) as f:
            coco = json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        counts = Counter(
            ann.get("category_id")
            for ann in annotations
        )

        print("  COCO: FOUND")
        print("  Images in JSON:", len(images))
        print("  Annotations:", len(annotations))

        print("  Categories:")

        for cat in categories:
            cid = cat.get("id")
            name = cat.get("name")

            print(
                f"    ID {cid}: {name} "
                f"({counts.get(cid, 0)} annotations)"
            )

        # Chá»‰ kiá»ƒm tra ID vÃ  file tham chiáº¿u
        image_ids = [im.get("id") for im in images]
        known_ids = set(image_ids)
        known_cats = {
            cat.get("id") for cat in categories
        }

        missing_refs = sum(
            ann.get("image_id") not in known_ids
            for ann in annotations
        )

        invalid_cats = sum(
            ann.get("category_id") not in known_cats
            for ann in annotations
        )

        print(
            "  Duplicate image IDs:",
            len(image_ids) - len(known_ids)
        )
        print(
            "  Missing image references:",
            missing_refs
        )
        print(
            "  Invalid category references:",
            invalid_cats
        )

    except (OSError, ValueError, TypeError) as e:
        print(
            "  COCO error:",
            type(e).__name__
        )


# ============================================================
# 4. TRAINED MODELS
# ============================================================

section("4. TRAINED MODEL ARTIFACTS")

if MODEL_FILES:
    for path in MODEL_FILES:
        print(
            f"{rel(path)} | "
            f"{size_mb(path)} MB"
        )
else:
    print("No model files found")

print("\nWeights loaded: NO")
print("Inference executed: NO")


# ============================================================
# 5. GIT - READ ONLY
# ============================================================

section("5. GIT REPOSITORY")

GIT_DIR = ROOT / ".git"


def git_readonly(args):
    try:
        env = os.environ.copy()
        env["GIT_OPTIONAL_LOCKS"] = "0"
        env["GIT_TERMINAL_PROMPT"] = "0"

        result = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False
        )

        if result.returncode != 0:
            return "Unavailable"

        return result.stdout.strip()

    except (
        FileNotFoundError,
        subprocess.TimeoutExpired
    ):
        return "Unavailable"


if GIT_DIR.exists():

    print("Git repository: FOUND")

    print(
        "Branch:",
        git_readonly([
            "rev-parse", "--abbrev-ref", "HEAD"
        ])
    )

    print(
        "Current commit:",
        git_readonly([
            "rev-parse", "--short", "HEAD"
        ])
    )

    print(
        "Remote names:",
        git_readonly(["remote"])
    )

    print("\nRecent commits:")

    print(
        git_readonly([
            "log", "-5",
            "--format=%h %as"
        ])
    )

    print(
        "\nGit status: NOT EXECUTED "
        "(avoid touching index)"
    )

else:
    print("Git repository: NOT FOUND")


# ============================================================
# 6. DVC
# ============================================================

section("6. DVC CONFIGURATION")

DVC_DIR = ROOT / ".dvc"

print("DVC initialized:", DVC_DIR.is_dir())

dvc_files = sorted(
    p for p in ALL_FILES
    if p.suffix.lower() == ".dvc"
)

print("\nTracked DVC files:")

if dvc_files:
    for p in dvc_files:
        print(" -", rel(p))
else:
    print(" None")

config_path = DVC_DIR / "config"

if config_path.is_file():

    parser = configparser.ConfigParser(
        interpolation=None
    )

    try:
        parser.read(
            config_path,
            encoding="utf-8"
        )

        print("\nDVC config sections:")

        for section_name in parser.sections():
            print(" -", section_name)

        remote_sections = [
            s for s in parser.sections()
            if s.startswith('remote "')
        ]

        print(
            "Configured remotes:",
            len(remote_sections)
        )

    except configparser.Error:
        print("Cannot parse DVC config")

print("\nDVC pipeline files:")

for name in ["dvc.yaml", "dvc.lock"]:
    if (ROOT / name).is_file():
        print(" -", name)

print("DVC pull/push/status: NOT EXECUTED")


# ============================================================
# 7. MLFLOW / DAGSHUB
# ============================================================

section("7. MLFLOW / DAGSHUB")

# TÃ¬m mlruns á»Ÿ cÃ¡c vá»‹ trÃ­ thá»±c táº¿,
# khÃ´ng Ä‘á»c artifact hoáº·c metadata nháº¡y cáº£m
mlflow_dirs = []
mlflow_dbs = []

for folder in [
    ROOT,
    ROOT / "mobilenet_model",
    ROOT / "resnet_model",
    ROOT / "cnn_model",
    ROOT / "api",
    ROOT / "dashboard"
]:
    if (folder / "mlruns").is_dir():
        mlflow_dirs.append(folder / "mlruns")

    if (folder / "mlflow.db").is_file():
        mlflow_dbs.append(folder / "mlflow.db")

print("MLflow directories:", len(mlflow_dirs))

for p in mlflow_dirs:
    print(" -", rel(p))

print("\nMLflow databases:", len(mlflow_dbs))

for p in mlflow_dbs:
    print(
        " -", rel(p),
        f"({size_mb(p)} MB)"
    )

print("\nMLflow / DagsHub integration scripts:")

integration_scripts = []

for path in PYTHON_FILES:
    source = read_source(path).lower()

    if "mlflow" in source or "dagshub" in source:
        integration_scripts.append(path)

for p in integration_scripts:
    print(" -", rel(p))

print(
    "\nModel Registry references:",
    sum(
        any(
            word in read_source(p).lower()
            for word in [
                "register_model",
                "models:/",
                "registered_model",
                "set_registered_model_alias"
            ]
        )
        for p in PYTHON_FILES
    ),
    "Python scripts"
)

print("Remote server connection: NOT TESTED")
print("API tokens: NOT READ")


# ============================================================
# 8. AUTOMATION / DEPLOYMENT
# ============================================================

section("8. AUTOMATION / DEPLOYMENT")

workflow_dir = ROOT / ".github" / "workflows"

workflows = []

if workflow_dir.is_dir():
    workflows = sorted(
        p.name for p in workflow_dir.iterdir()
        if p.suffix.lower() in {".yaml", ".yml"}
    )

print("GitHub Actions:")

if workflows:
    for name in workflows:
        print(" -", name)
else:
    print(" NOT FOUND")

docker_files = [
    p for p in ALL_FILES
    if p.name.lower() in {
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml"
    }
]

print("\nDocker:")

if docker_files:
    for p in docker_files:
        print(" -", rel(p))
else:
    print(" NOT FOUND")

training_scripts = [
    p for p in PYTHON_FILES
    if any(
        keyword in p.name.lower()
        for keyword in [
            "train", "retrain", "fine_tune"
        ]
    )
    and p.name != Path(__file__).name
]

print("\nTraining / Retraining scripts:")

if training_scripts:
    for p in training_scripts:
        print(" -", rel(p))
else:
    print(" NOT FOUND")

kaggle_scripts = [
    p for p in PYTHON_FILES
    if "kaggle" in read_source(p).lower()
    and p.name != Path(__file__).name
]

print("\nScripts referencing Kaggle:")

if kaggle_scripts:
    for p in kaggle_scripts:
        print(" -", rel(p))
else:
    print(" NOT FOUND")

print("\nKaggle API: NOT CALLED")


# ============================================================
# 9. MLOPS READINESS SUMMARY
# ============================================================

section("9. MLOPS READINESS SUMMARY")

CHECKS = {
    "PKLot Train COCO": (
        DATA_DIR / "train" /
        "_annotations.coco.json"
    ).is_file(),

    "PKLot Valid COCO": (
        DATA_DIR / "valid" /
        "_annotations.coco.json"
    ).is_file(),

    "PKLot Test COCO": (
        DATA_DIR / "test" /
        "_annotations.coco.json"
    ).is_file(),

    "Online Stream Folder": (
        DATA_DIR / "realtime_stream"
    ).is_dir(),

    "Online Stream COCO": (
        DATA_DIR / "realtime_stream" /
        "_annotations.coco.json"
    ).is_file(),

    "Confirmed Data Folder": (
        DATA_DIR / "confirmed"
    ).is_dir(),

    "MobileNetV3 Weights": (
        ROOT / "mobilenet_model" /
        "mobilenet_v3_small_best.pth"
    ).is_file(),

    "ResNet18 Weights": (
        ROOT / "resnet_model" /
        "resnet18_best.pth"
    ).is_file(),

    "Custom CNN Weights": (
        ROOT / "cnn_model" /
        "optimized_custom_cnn_best.pth"
    ).is_file(),

    "FastAPI Source": (
        ROOT / "api" / "main.py"
    ).is_file(),

    "Streamlit Source": (
        ROOT / "dashboard" / "app.py"
    ).is_file(),

    "DVC Initialized": DVC_DIR.is_dir(),

    "DVC Data Tracking": (
        ROOT / "data.dvc"
    ).is_file(),

    "Git Repository": GIT_DIR.exists(),

    "MLflow Local Artifacts": bool(
        mlflow_dirs or mlflow_dbs
    ),

    "MLflow Integration Code": bool(
        integration_scripts
    ),

    "GitHub Actions": bool(workflows),

    "Docker Configuration": bool(docker_files),

    "Retraining Script": bool(
        training_scripts
    )
}

found_count = 0

for name, exists in CHECKS.items():
    status = "FOUND" if exists else "NOT FOUND"

    if exists:
        found_count += 1

    print(f"[{status}] {name}")

print("\nDetected components:",
      f"{found_count}/{len(CHECKS)}")

print(
    "Note: FOUND means presence only, "
    "not verified functionality."
)


# ============================================================
# COMPLETE
# ============================================================

section("AUDIT COMPLETE - READ ONLY")

print("Project files modified: NO")
print("Report files created: NO")
print("Models loaded: NO")
print("Training executed: NO")
print("DVC push/pull executed: NO")
print("Git commits created: NO")
print("Remote API calls: NO")
print("Credentials displayed: NO")
print("\nAudit finished successfully.")
