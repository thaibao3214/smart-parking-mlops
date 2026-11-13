"""Stage 2 smoke-test: create a small, class-balanced crop dataset from PKLot train/valid only.
Never includes test or confirmed_online. Requires local data to have been pulled by DVC.
"""
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.stage2_dataset'
RNG = random.Random(42)


def build(split: str, limit_per_class: int):
    source = ROOT / 'data' / split
    coco = json.loads((source / '_annotations.coco.json').read_text(encoding='utf-8'))
    images = {im['id']: im for im in coco['images']}
    buckets = defaultdict(list)
    for ann in coco['annotations']:
        category = ann.get('category_id')
        if category not in (1, 2):
            continue
        im = images.get(ann.get('image_id'))
        if not im:
            continue
        path = source / Path(im['file_name']).name
        if not path.is_file():
            continue
        x, y, w, h = ann['bbox']
        if not all(math.isfinite(float(v)) for v in (x, y, w, h)) or w <= 5 or h <= 5:
            continue
        buckets[category - 1].append((path, (x, y, w, h), ann['id']))
    dest = OUT / split
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in (0, 1):
        entries = buckets[label]
        RNG.shuffle(entries)
        # Iterate beyond initial picks if a crop happens to be invalid.
        for path, (x, y, w, h), ann_id in entries:
            if sum(r['label'] == label for r in rows) >= limit_per_class:
                break
            with Image.open(path) as img:
                rgb = img.convert('RGB')
                x1, y1 = max(0, math.floor(x)), max(0, math.floor(y))
                x2, y2 = min(rgb.width, math.ceil(x + w)), min(rgb.height, math.ceil(y + h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = rgb.crop((x1, y1, x2, y2)).resize((64, 64))
                filename = f'{label}_{len(rows):04d}_{ann_id}.png'
                crop.save(dest / filename)
            rows.append({'file': f'{split}/{filename}', 'label': label})
        if sum(r['label'] == label for r in rows) != limit_per_class:
            raise RuntimeError(f'{split}: not enough valid class {label} crops')
    return rows


def main():
    if OUT.exists():
        raise RuntimeError(f'{OUT} already exists; remove only this staging directory first')
    OUT.mkdir()
    train = build('train', 64)  # 128 crops
    valid = build('valid', 32)  # 64 crops
    pointer = (ROOT / 'data.dvc').read_bytes()
    meta = {
        'stage': 'stage2-smoke',
        'source': 'PKLot train/valid ONLY',
        'training_approved_for_production': False,
        'online_crops_included': 0,
        'test_crops_included': 0,
        'git_commit': os.environ.get('GITHUB_SHA', 'unknown'),
        'github_run_id': os.environ.get('GITHUB_RUN_ID', 'unknown'),
        'data_dvc_sha256': hashlib.sha256(pointer).hexdigest(),
        'train': train,
        'valid': valid,
    }
    (OUT / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    (OUT / 'dataset-metadata.json').write_text(json.dumps({
        'title': 'Smart Parking Stage2 Smoke Data',
        'id': 'nguynngthibo/smart-parking-stage2-smoke',
        'licenses': [{'name': 'unknown'}],
    }, indent=2), encoding='utf-8')
    print(f'STAGE2_PREP_PASS train={len(train)} valid={len(valid)}')


if __name__ == '__main__':
    main()
