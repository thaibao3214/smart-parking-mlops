
import json
from pathlib import Path
from collections import Counter

from PIL import Image

ROOT = Path("data/confirmed_online/frames")

label_counts = Counter()
prediction_counts = Counter()
errors = []
total_slots = 0
corrected = 0

files = sorted(ROOT.glob("*/labels.json"))

for json_path in files:
    frame_dir = json_path.parent

    try:
        data = json.loads(
            json_path.read_text(encoding="utf-8")
        )
    except Exception as e:
        errors.append(f"{json_path}: {e}")
        continue

    slots = data.get("slots", [])

    if not slots:
        errors.append(f"{json_path}: Khong co slots")

    for slot in slots:
        total_slots += 1

        confirmed = slot.get("confirmed_label")
        predicted = slot.get("predicted_label")
        label_id = slot.get("label_id")
        crop_file = slot.get("crop_file")

        if confirmed not in ("Empty", "Occupied"):
            errors.append(f"{json_path}: Nhan khong hop le")
            continue

        expected_id = 0 if confirmed == "Empty" else 1

        if label_id != expected_id:
            errors.append(
                f"{json_path}: label_id khong khop"
            )

        label_counts[confirmed] += 1
        prediction_counts[predicted] += 1

        if confirmed != predicted:
            corrected += 1

        if not isinstance(crop_file, str):
            errors.append(f"{json_path}: crop_file khong hop le")
            continue

        crop_path = frame_dir / crop_file

        if not crop_path.is_file():
            errors.append(f"Thieu crop: {crop_path}")
            continue

        try:
            with Image.open(crop_path) as image:
                if image.size != (64, 64):
                    errors.append(
                        f"Sai kich thuoc: {crop_path}"
                    )

                if image.mode != "RGB":
                    errors.append(
                        f"Sai RGB: {crop_path}"
                    )

                image.verify()
        except Exception as e:
            errors.append(f"Loi crop {crop_path}: {e}")


print("\n========== ONLINE DATA VALIDATION ==========")
print("Frames:", len(files))
print("Total slots:", total_slots)
print("Confirmed Empty:", label_counts["Empty"])
print("Confirmed Occupied:", label_counts["Occupied"])
print("AI predictions:", dict(prediction_counts))
print("Human corrections:", corrected)
print("Errors:", len(errors))

if errors:
    print("\nCHI TIET LOI:")
    for error in errors[:30]:
        print("-", error)
else:
    print("\nVALIDATION PASSED")

print("============================================")
