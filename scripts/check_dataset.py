
import json
import random
from pathlib import Path
from collections import Counter

import requests
from PIL import Image

# =========================
# CAU HINH
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "data" / "test"
JSON_PATH = TEST_DIR / "_annotations.coco.json"

API_URL = "http://127.0.0.1:8000/predict"

NUM_IMAGES = 5
random.seed(42)

# =========================
# DOC COCO JSON
# =========================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

categories = {
    c["id"]: c["name"]
    for c in coco["categories"]
}

images = {
    img["id"]: img
    for img in coco["images"]
}

annotations = {}

for ann in coco["annotations"]:
    annotations.setdefault(
        ann["image_id"], []
    ).append(ann)

print("\n========== 1. DATASET ==========")

print("So anh:", len(images))
print("So annotations:", len(coco["annotations"]))
print("Categories:", categories)

category_counts = Counter(
    ann["category_id"]
    for ann in coco["annotations"]
)

for cid, count in category_counts.items():
    print(f"{categories.get(cid, cid)}: {count}")

# =========================
# XAC DINH NHAN
# =========================
empty_ids = {
    cid for cid, name in categories.items()
    if name.lower() in ("space-empty", "empty")
}

occupied_ids = {
    cid for cid, name in categories.items()
    if name.lower() in ("space-occupied", "occupied")
}

if not empty_ids or not occupied_ids:
    raise ValueError(
        "Khong tim thay nhan Empty/Occupied trong COCO"
    )

# =========================
# CHON ANH TEST
# =========================
candidates = []

for img_id, img_info in images.items():

    path = TEST_DIR / img_info["file_name"]

    if not path.exists():
        continue

    anns = annotations.get(img_id, [])

    n_empty = sum(
        a["category_id"] in empty_ids
        for a in anns
    )

    n_occupied = sum(
        a["category_id"] in occupied_ids
        for a in anns
    )

    # Uu tien anh co ca 2 lop
    if n_empty > 0 and n_occupied > 0:
        candidates.append((img_id, path))

random.shuffle(candidates)

selected = candidates[:NUM_IMAGES]

print("\n========== 2. KIEM TRA ANH ==========")
print("So anh co ca Empty va Occupied:", len(candidates))

if not selected:
    raise RuntimeError(
        "Khong co anh hop le de kiem tra"
    )

total_correct = 0
total_compared = 0
total_empty_pred = 0
total_occupied_pred = 0
total_empty_true = 0
total_occupied_true = 0

# =========================
# KIEM TRA TUNG ANH
# =========================
for img_id, path in selected:

    info = images[img_id]
    anns = annotations[img_id]

    with Image.open(path) as image:
        original_size = image.size

    print("\n--------------------------------")
    print("Anh:", path.name)
    print("Kich thuoc file:", original_size)
    print(
        "Kich thuoc COCO:",
        info.get("width"),
        "x",
        info.get("height")
    )

    if original_size != (640, 640):
        print(
            "CANH BAO: Anh khong phai 640x640."
            " Can kiem tra scale bbox khi resize."
        )

    # Nhan thuc te
    true_empty = sum(
        a["category_id"] in empty_ids
        for a in anns
    )

    true_occupied = sum(
        a["category_id"] in occupied_ids
        for a in anns
    )

    other = len(anns) - true_empty - true_occupied

    print("Ground Truth:")
    print("  Empty:", true_empty)
    print("  Occupied:", true_occupied)
    print("  Other annotations:", other)

    total_empty_true += true_empty
    total_occupied_true += true_occupied

    try:
        with open(path, "rb") as f:
            response = requests.post(
                API_URL,
                data={"model_type": "mobilenetv3"},
                files={
                    "file": (
                        path.name,
                        f,
                        "image/jpeg"
                    )
                },
                timeout=180
            )

        response.raise_for_status()
        result = response.json()

    except Exception as e:
        print("API ERROR:", e)
        continue

    print("Prediction:")
    print("  Empty:", result["empty_spaces"])
    print("  Occupied:", result["occupied_spaces"])
    print("  Model source:", result.get("model_source"))
    print("  Model version:", result.get("model_version"))

    total_empty_pred += result["empty_spaces"]
    total_occupied_pred += result["occupied_spaces"]

    # API hien tai gan id theo thu tu annotation
    correct = 0
    compared = 0

    for pred in result["predictions"]:

        idx = pred["id"] - 1

        if idx >= len(anns):
            continue

        cid = anns[idx]["category_id"]

        if cid in empty_ids:
            true_label = "Empty"
        elif cid in occupied_ids:
            true_label = "Occupied"
        else:
            continue

        pred_label = pred["status"]

        compared += 1

        if pred_label.lower() == true_label.lower():
            correct += 1

    if compared:
        accuracy = 100 * correct / compared
        print(
            f"Accuracy anh: {accuracy:.2f}%"
            f" ({correct}/{compared})"
        )

        total_correct += correct
        total_compared += compared

# =========================
# TONG KET
# =========================
print("\n========== 3. TONG KET ==========")

print("Ground Truth Empty:", total_empty_true)
print("Ground Truth Occupied:", total_occupied_true)

print("Predicted Empty:", total_empty_pred)
print("Predicted Occupied:", total_occupied_pred)

if total_compared:
    print(
        "Accuracy tren cac o da doi chieu:",
        f"{100 * total_correct / total_compared:.2f}%"
    )
    print(
        "So o duoc doi chieu:",
        total_compared
    )
else:
    print("Khong co du lieu du de tinh accuracy")

if total_occupied_true > 0 and total_occupied_pred == 0:
    print(
        "\nCANH BAO: Du lieu co xe nhung "
        "model du doan tat ca la Empty."
    )

if total_compared:
    print(
        "\nLuu y: Accuracy nay chi la kiem tra "
        "nhanh tren anh mau, khong phai test "
        "accuracy chinh thuc."
    )

print("\nCHECK DATASET COMPLETED")
