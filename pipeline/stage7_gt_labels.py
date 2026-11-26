"""Strict provenance/geometry check for automated PKLot COCO TRAIN labels.

Never call this mode 'manual_frame_review'.  All crops/labels must match the
original COCO TRAIN annotations, independently of model predictions.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageChops

GT_METHOD = 'pklot_coco_ground_truth'
LABELS = {1: ('Empty', 0), 2: ('Occupied', 1)}


@lru_cache(maxsize=4)
def coco_for(path: str):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    images = {str(img['id']): img for img in doc['images']}
    annotations = {}
    for ann in doc['annotations']:
        if int(ann.get('category_id', -1)) in LABELS:
            annotations.setdefault(str(ann['image_id']), []).append(ann)
    for rows in annotations.values():
        rows.sort(key=lambda a: int(a['id']))
    return images, annotations


def coordinates(bbox, size):
    x, y, w, h = map(float, bbox)
    if w <= 0 or h <= 0:
        raise ValueError('Invalid ground truth bbox size')
    width, height = size
    x0 = max(0, min(width, round(x)))
    y0 = max(0, min(height, round(y)))
    x1 = max(0, min(width, round(x + w)))
    y1 = max(0, min(height, round(y + h)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Ground truth bbox outside image')
    return [x0, y0, x1 - x0, y1 - y0]


def expected_annotations(root: Path, entry):
    images, all_anns = coco_for(str(root / 'data' / 'train' / '_annotations.coco.json'))
    image_id = str(entry['image_id'])
    image_meta = images.get(image_id)
    if not image_meta or Path(image_meta['file_name']).name != entry['source_file_name']:
        raise ValueError('COCO image/source mismatch')
    annotations = all_anns.get(image_id, [])
    if len(annotations) != 100 or len({int(a['id']) for a in annotations}) != 100:
        raise ValueError('Require 100 unique Empty/Occupied COCO annotations')
    return annotations


def validate_gt_record(root: Path, sim: Path, conf: Path, entry, payload, labels_path: Path):
    """Fail closed unless a saved auto-confirmed frame is pixel-equal to COCO TRAIN GT."""
    root, sim, conf = Path(root), Path(sim), Path(conf)
    if not sim.is_absolute():
        sim = root / sim
    if not conf.is_absolute():
        conf = root / conf
    if payload.get('review_method') != GT_METHOD:
        raise ValueError('Unexpected review method in GT verifier')
    if (payload.get('test_only') is not True or payload.get('production_approved') is not False
            or payload.get('source_split') != 'data/train'
            or payload.get('origin') != 'pklot_train_simulation'
            or str(payload.get('source_image_id')) != str(entry['image_id'])
            or payload.get('source_image') != entry['file_name']
            or payload.get('source_file_name') != entry['source_file_name']):
        raise ValueError('Invalid GT provenance/source reference')
    source = root / 'data' / 'train' / entry['source_file_name']
    from hashlib import sha256
    if (not source.is_file() or not (sim / 'images' / entry['file_name']).is_file()
            or sha256(source.read_bytes()).digest() != sha256((sim / 'images' / entry['file_name']).read_bytes()).digest()):
        raise ValueError('Source image/stream content mismatch')
    actual = payload.get('slots')
    annotations = expected_annotations(root, entry)
    if not isinstance(actual, list) or len(actual) != len(annotations):
        raise ValueError('Missing/extra GT slots')
    with Image.open(source) as im:
        image = im.convert('RGB')
        for index, (annotation, slot) in enumerate(zip(annotations, actual), 1):
            label, label_id = LABELS[int(annotation['category_id'])]
            bbox = coordinates(annotation['bbox'], image.size)
            crop_name = f'slot_{index:03d}.png'
            if (int(slot['slot_id']) != index or slot.get('crop_file') != crop_name
                    or slot.get('confirmed_label') != label or int(slot.get('label_id', -1)) != label_id
                    or slot.get('bbox_xywh') != bbox
                    or int(slot.get('source_annotation_id', -1)) != int(annotation['id'])):
                raise ValueError(f'GT label/bbox/annotation mismatch for slot {index}')
            x, y, w, h = bbox
            expected = image.crop((x, y, x+w, y+h)).resize((64, 64))
            stored_path = labels_path.parent / crop_name
            if not stored_path.is_file():
                raise ValueError(f'Missing GT crop: {crop_name}')
            with Image.open(stored_path) as saved:
                if saved.size != (64, 64) or ImageChops.difference(
                        expected, saved.convert('RGB')).getbbox() is not None:
                    raise ValueError(f'Ground truth crop pixels mismatch: {crop_name}')
    return True
