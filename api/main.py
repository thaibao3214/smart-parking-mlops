
import io
import hashlib
import json
import os
import time
import threading
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, resnet18
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

import dagshub
import mlflow
from mlflow.tracking import MlflowClient


# ==========================================
# 1. CAU HINH
# ==========================================
# Works with main.py in either project/ or project/backend/.
_THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = next(
    (p for p in (_THIS_DIR, _THIS_DIR.parent)
     if (p / "data" / "test" / "_annotations.coco.json").is_file()
     or (p / "mobilenet_model").is_dir()),
    _THIS_DIR,
)
DEVICE = torch.device("cpu")

MODEL_NAME = "SmartParking-MobileNetV3-Small"
MODEL_ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

CHECK_INTERVAL = 60
BATCH_SIZE = 1

COCO_SPLITS = ("test", "train")
SIMULATION_MANIFESTS = (
    BASE_DIR / "simulation_train" / "manifest.json",
    BASE_DIR / "data" / "realtime_stream" / "manifest.json",
)

PATH_MAP = {
    "mobilenetv3": BASE_DIR / "mobilenet_model" / "mobilenet_v3_small_best.pth",
    "resnet18": BASE_DIR / "resnet_model" / "resnet18_best.pth",
    "custom_cnn": BASE_DIR / "cnn_model" / "optimized_custom_cnn_best.pth",
}

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 2. CUSTOM CNN
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)


class OptimizedPKLotCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        return self.classifier(x)


# ==========================================
# 3. LOAD MODEL LOCAL
# ==========================================
def load_local_model(model_type):
    if model_type == "mobilenetv3":
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, 2
        )

    elif model_type == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)

    elif model_type == "custom_cnn":
        model = OptimizedPKLotCNN(2)

    else:
        raise ValueError("Model type khong hop le")

    path = PATH_MAP[model_type]

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


# ==========================================
# 4. MLFLOW MODEL MANAGER
# ==========================================
class ModelManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = None
        self.version = None
        self.source = "uninitialized"
        self.last_check = 0
        self.client = None
        self.other_models = {}

    def connect(self):
        dagshub.init(
            repo_owner="thaibao3214",
            repo_name="smart-parking-mlops",
            mlflow=True
        )
        self.client = MlflowClient()

    def load_registry_model(self, force=False):
        now = time.monotonic()

        with self.lock:
            if not force and now - self.last_check < CHECK_INTERVAL:
                return

            self.last_check = now

            try:
                if self.client is None:
                    self.connect()

                info = self.client.get_model_version_by_alias(
                    MODEL_NAME, MODEL_ALIAS
                )

                if self.model is not None and self.version == info.version:
                    return

                print(f"Loading production version {info.version}...")

                new_model = mlflow.pytorch.load_model(
                    f"models:/{MODEL_NAME}/{info.version}",
                    map_location="cpu"
                )

                # Exported GraphModule khong goi eval()
                with torch.no_grad():
                    result = new_model(torch.zeros(1, 3, 64, 64))

                if result.shape != (1, 2):
                    raise ValueError("Model output shape khong hop le")

                self.model = new_model
                self.version = info.version
                self.source = "DagsHub Registry"

                print(f"Production model ready: Version {self.version}")

            except Exception as e:
                print("Registry error:", e)

                if self.model is None:
                    try:
                        self.model = load_local_model("mobilenetv3")
                        self.version = "local"
                        self.source = "Local fallback"
                        print("Using local MobileNetV3 fallback")
                    except Exception as local_error:
                        print("Local fallback error:", local_error)

    def get_model(self, model_type):
        if model_type == "mobilenetv3":
            self.load_registry_model()

            with self.lock:
                if self.model is None:
                    raise RuntimeError("MobileNetV3 chua san sang")

                return self.model, self.version, self.source

        with self.lock:
            if model_type not in self.other_models:
                self.other_models[model_type] = load_local_model(model_type)

            return self.other_models[model_type], "local", "Local weights"


manager = ModelManager()


# ==========================================
# 5. COCO INDEX: TEST + TRAIN + SIMULATIONS
# ==========================================
def load_coco_index():
    """Index only ROIs (bounding boxes); NEVER use COCO class labels.

    Filename may repeat across train/test. Therefore each candidate stores its
    own file path; /predict verifies raw file SHA256 before using its boxes.
    """
    index = {}
    source_lookup = {}

    for split in COCO_SPLITS:
        directory = BASE_DIR / "data" / split
        annotation_path = directory / "_annotations.coco.json"
        if not annotation_path.is_file():
            print("COCO missing:", annotation_path)
            continue
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = {int(row["id"]): row for row in coco.get("images", [])}
        grouped = {}
        for ann in coco.get("annotations", []):
            if int(ann.get("category_id", -1)) not in (1, 2):
                continue
            grouped.setdefault(int(ann["image_id"]), []).append(ann["bbox"])

        for image_id, image_info in images.items():
            filename = Path(image_info["file_name"]).name
            path = directory / filename
            if not path.is_file():
                continue
            bboxes = grouped.get(image_id, [])
            if not bboxes:
                continue
            row = {
                "path": path,
                "source_split": f"data/{split}",
                "source_image": filename,
                "bboxes": bboxes,
                "width": int(image_info.get("width") or 640),
                "height": int(image_info.get("height") or 640),
            }
            index.setdefault(filename, []).append(row)
            source_lookup[(split, filename)] = row

    # Generated simulation filenames (frame_001.jpg etc.) are not COCO names.
    # Map them through the simulation manifest to the ORIGINAL train/test ROIs.
    for manifest_path in list(SIMULATION_MANIFESTS) + sorted((BASE_DIR / 'simulation_batches').glob('*/manifest.json')):
        if not manifest_path.is_file():
            continue
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = doc.get("source_split")
        if split not in ("data/train", "data/test"):
            # The historical manifest may not specify top-level source_split.
            split = "data/test" if "realtime_stream" in str(manifest_path) else None
        for entry in doc.get("images", []):
            alias = Path(str(entry.get("file_name", ""))).name
            original = Path(str(entry.get("source_file_name", alias))).name
            row_split = entry.get("source_split", split)
            if row_split not in ("data/train", "data/test"):
                continue
            source = source_lookup.get((row_split.split("/")[-1], original))
            sim_file = manifest_path.parent / "images" / alias
            if source is None or not sim_file.is_file():
                continue
            # Prefer manifest mapping for generated filenames. Match by content
            # hash at request time, so a different image cannot borrow ROIs.
            index.setdefault(alias, []).append({
                **source,
                "path": sim_file,
            })
    return index


coco_index = {}


def resolve_image_annotations(filename, image_bytes):
    candidates = coco_index.get(filename, [])
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No COCO ROI mapping for {filename}. Only indexed PKLot images are supported.",
        )

    actual_sha = hashlib.sha256(image_bytes).hexdigest()
    for row in candidates:
        if hashlib.sha256(row["path"].read_bytes()).hexdigest() == actual_sha:
            return row

    raise HTTPException(
        status_code=422,
        detail=("Uploaded image does not match its indexed PKLot frame. "
                "Check the simulation manifest and original filename."),
    )


@asynccontextmanager
async def lifespan(app):
    global coco_index
    coco_index = load_coco_index()
    print("COCO image names indexed:", len(coco_index))
    manager.load_registry_model(force=True)
    yield


app = FastAPI(
    title="Smart Parking MLOps PKLot API",
    version="4.1",
    lifespan=lifespan,
)


# ==========================================
# 6. API STATUS
# ==========================================
@app.get("/")
def home():
    return {
        "message": "Smart Parking MLOps API is running",
        "model": MODEL_NAME
    }


@app.get("/health")
def health():
    with manager.lock:
        return {
            "status": "healthy" if manager.model is not None else "degraded",
            "model_source": manager.source,
            "model_version": manager.version,
            "device": str(DEVICE),
            "registry_alias": MODEL_ALIAS
        }


@app.post("/reload-model")
def reload_model():
    manager.load_registry_model(force=True)
    return health()


# ==========================================
# 7. PREDICT
# ==========================================
@app.post("/predict")
async def predict_parking(
    model_type: str = Form("mobilenetv3"),
    file: UploadFile = File(...),
):
    if model_type not in PATH_MAP:
        raise HTTPException(status_code=400, detail="Invalid model type")

    image_name = Path(file.filename or "").name
    if not image_name:
        raise HTTPException(status_code=400, detail="Missing image filename")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise ValueError("Empty image upload")
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_pil = image.convert("RGB")
        image_pil = image_pil.resize((640, 640))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    # Only image bytes matching the indexed PKLot frames can receive COCO ROIs.
    # No Empty/Occupied ground-truth label is read during prediction.
    annotation = resolve_image_annotations(image_name, image_bytes)

    try:
        model, version, source = manager.get_model(model_type)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    valid_boxes = []
    tensors = []
    # COCO bbox dimensions are in the original annotation coordinate space.
    scale_x = 640 / annotation["width"]
    scale_y = 640 / annotation["height"]
    for idx, (x, y, w, h) in enumerate(annotation["bboxes"], start=1):
        xmin = max(0, int(x * scale_x))
        ymin = max(0, int(y * scale_y))
        xmax = min(640, int((x + w) * scale_x))
        ymax = min(640, int((y + h) * scale_y))
        if xmax - xmin <= 5 or ymax - ymin <= 5:
            continue
        crop = image_pil.crop((xmin, ymin, xmax, ymax))
        tensors.append(transform(crop))
        valid_boxes.append({
            "id": idx,
            "box": [xmin, ymin, xmax - xmin, ymax - ymin],
        })

    predictions = []
    count_empty = 0
    count_occupied = 0
    with torch.inference_mode():
        for start in range(0, len(tensors), BATCH_SIZE):
            batch = torch.stack(tensors[start:start + BATCH_SIZE]).to(DEVICE)
            outputs = model(batch)
            labels = outputs.argmax(dim=1).tolist()
            for box_info, label in zip(
                valid_boxes[start:start + BATCH_SIZE], labels
            ):
                status = "Occupied" if label == 1 else "Empty"
                if label == 1:
                    count_occupied += 1
                else:
                    count_empty += 1
                predictions.append({**box_info, "status": status})

    total = count_empty + count_occupied
    rate = 100 * count_occupied / total if total else 0
    return {
        "model_used": model_type,
        "model_version": version,
        "model_source": source,
        "image_name": image_name,
        "source_split": annotation["source_split"],
        "total_spaces": total,
        "occupied_spaces": count_occupied,
        "empty_spaces": count_empty,
        "occupancy_rate": round(rate, 2),
        "predictions": predictions,
    }
