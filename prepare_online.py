
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "data" / "test"
TEST_JSON = TEST_DIR / "_annotations.coco.json"
OUTPUT_DIR = BASE_DIR / "data" / "realtime_stream"


def layout_signature(annotations, cell):
    """
    Gom ảnh theo bố cục ROI gần giống nhau.
    Chỉ dùng bbox, KHÔNG dùng nhãn Empty/Occupied.
    """
    centers = []

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cx = x + w / 2
        cy = y + h / 2

        centers.append((
            round(cx / cell),
            round(cy / cell)
        ))

    return (
        len(annotations),
        tuple(sorted(centers))
    )


def prepare(count=20, seed=42, cell=32, dry_run=False):
    if not TEST_JSON.is_file():
        raise FileNotFoundError(TEST_JSON)

    with TEST_JSON.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    annotations_by_image = defaultdict(list)

    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    groups = defaultdict(list)

    for image in coco["images"]:
        name = Path(image["file_name"]).name
        image_path = TEST_DIR / name

        if not image_path.is_file():
            continue

        anns = annotations_by_image.get(image["id"], [])

        if not anns:
            continue

        signature = layout_signature(anns, cell)
        groups[signature].append(image)

    candidates = [
        images
        for images in groups.values()
        if len(images) >= count
    ]

    if not candidates:
        largest = max(
            (len(g) for g in groups.values()),
            default=0
        )

        raise RuntimeError(
            f"Không có nhóm đủ {count} ảnh. "
            f"Nhóm lớn nhất có {largest} ảnh. "
            "Thử --cell 48 hoặc --cell 64."
        )

    # Ưu tiên nhóm có nhiều ảnh và bố cục giống nhau
    selected_group = max(candidates, key=len)

    rng = random.Random(seed)
    selected = rng.sample(selected_group, count)

    # Phát ảnh theo tên (thường chứa timestamp PKLot)
    selected.sort(key=lambda img: img["file_name"])

    selected_ids = {img["id"] for img in selected}

    selected_annotations = [
        ann for ann in coco["annotations"]
        if ann["image_id"] in selected_ids
    ]

    selected_images = []

    for image in selected:
        new_image = image.copy()
        name = Path(image["file_name"]).name

        new_image["file_name"] = f"images/{name}"
        selected_images.append(new_image)

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": selected_images,
        "annotations": selected_annotations
    }

    manifest = {
        "count": len(selected),
        "seed": seed,
        "layout_cell": cell,
        "source": "data/test",
        "images": [
            {
                "file_name": Path(img["file_name"]).name,
                "image_id": img["id"]
            }
            for img in selected
        ]
    }

    print("\n========== ONLINE DATA ==========")
    print("Số ảnh:", len(selected))
    print("Số annotations:", len(selected_annotations))
    print("Nhóm bố cục:", len(selected_group), "ảnh")
    print("Chế độ:", "DRY RUN" if dry_run else "CREATE")

    for i, image in enumerate(selected, 1):
        print(f"{i:02d}. {Path(image['file_name']).name}")

    if dry_run:
        print("\nKhông có file nào được thay đổi.")
        return

    images_dir = OUTPUT_DIR / "images"

    # Không ghi đè nếu đã chuẩn bị dữ liệu trước đó
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} đã có dữ liệu. "
            "Không tự động ghi đè."
        )

    images_dir.mkdir(parents=True, exist_ok=True)

    for image in selected:
        filename = Path(image["file_name"]).name

        shutil.copy2(
            TEST_DIR / filename,
            images_dir / filename
        )

    coco_path = OUTPUT_DIR / "_annotations.coco.json"

    with coco_path.open("w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False, indent=2)

    manifest_path = OUTPUT_DIR / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nĐã tạo thành công:")
    print("Images:", images_dir)
    print("COCO:", coco_path)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.count < 1 or args.cell < 1:
        parser.error("--count và --cell phải lớn hơn 0")

    prepare(
        count=args.count,
        seed=args.seed,
        cell=args.cell,
        dry_run=args.dry_run
    )
