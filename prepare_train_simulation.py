"""Create an immutable PKLot train TEST_ONLY simulation in simulation_batches/<batch>.
Usage: python prepare_train_simulation.py --batch batch_002 --count 20 --seed 2026
"""
import argparse
import json
import random
import re
import shutil
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / 'data' / 'train'


def signature(anns, cell):
    centres = []
    for a in anns:
        if int(a.get('category_id', -1)) not in (1, 2):
            continue
        x, y, w, h = a['bbox']
        centres.append((round((x + w / 2) / cell), round((y + h / 2) / cell)))
    return len(centres), tuple(sorted(centres))


def used_originals():
    previous = [ROOT / 'simulation_train' / 'manifest.json']
    previous.extend(sorted((ROOT / 'simulation_batches').glob('*/manifest.json')))
    used = set()
    for path in previous:
        if path.exists():
            m = json.loads(path.read_text(encoding='utf-8'))
            if m.get('source_split') == 'data/train':
                used.update(str(row.get('source_file_name', '')) for row in m.get('images', []))
    return used


def make_batch(batch, count, seed, cell):
    if not re.fullmatch(r'batch_[0-9]{3,6}', batch):
        raise ValueError('Batch must look like batch_002')
    if count != 20:
        raise ValueError('Stage7 TEST_ONLY requires exactly 20 frames')
    coco = json.loads((TRAIN / '_annotations.coco.json').read_text(encoding='utf-8'))
    anns = defaultdict(list)
    for a in coco['annotations']:
        anns[a['image_id']].append(a)
    used = used_originals()
    grouped = defaultdict(list)
    for image in coco['images']:
        filename = Path(image['file_name']).name
        if filename in used or not (TRAIN / filename).is_file():
            continue
        sig = signature(anns[image['id']], cell)
        if sig[0] == 100:
            grouped[sig].append(image)
    options = [g for g in grouped.values() if len(g) >= count]
    if not options:
        raise RuntimeError(f'No 100-space layout group has {count} unused frames. Try --cell 48 or --cell 64')
    group = max(options, key=len)
    selected = random.Random(seed).sample(group, count)
    dest = ROOT / 'simulation_batches' / batch
    if dest.exists():
        raise FileExistsError(f'Batch already exists, refusing overwrite: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / ('.building_' + uuid.uuid4().hex)
    try:
        (tmp / 'images').mkdir(parents=True)
        (tmp / 'predictions').mkdir()
        entries = []
        for i, image in enumerate(selected, 1):
            src = Path(image['file_name']).name
            target = f'frame_{i:03d}{Path(src).suffix.lower()}'
            shutil.copy2(TRAIN / src, tmp / 'images' / target)
            entries.append({'image_id': image['id'], 'file_name': target,
                            'source_file_name': src, 'source_split': 'data/train'})
        manifest = {'schema_version': 1, 'batch_name': batch,
                    'origin': 'pklot_train_simulation', 'source_split': 'data/train',
                    'test_only': True, 'production_approved': False,
                    'interval_seconds': 5, 'frame_count': count, 'images': entries}
        (tmp / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.rename(dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(f'STAGE7_BATCH_PREP_PASS: {batch}, {len(entries)} NEW train frames')
    print('Confirmed crop destination:', ROOT / 'confirmed_batches' / batch)
    print('TEST_ONLY_NO_PROMOTION')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--batch', default='batch_002')
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--cell', type=int, default=32)
    a = p.parse_args()
    make_batch(a.batch, a.count, a.seed, a.cell)
