"""Stage 7: fail-closed approval validation and per-run Kaggle payload packaging.

Approved crops must be from newly captured external camera frames (not PKLot splits).
No automatic use of data/realtime_stream or data/test.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from PIL import Image

CHECKPOINTS = {
    'mobilenetv3': 'mobilenet_model/mobilenet_v3_small_best.pth',
    'resnet18': 'resnet_model/resnet18_best.pth',
    'custom_cnn': 'cnn_model/optimized_custom_cnn_best.pth',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError(f'Invalid relative path {name!r}')
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
        raise ValueError(f'Unsafe path {name!r}')
    return path


def approved_samples(approved_dir, minimum):
    mf = approved_dir / 'manifest.json'
    if not mf.is_file():
        raise ValueError(f'Missing human-reviewed manifest: {mf}')
    doc = json.loads(mf.read_text(encoding='utf-8'))
    if doc.get('schema_version') != 1 or doc.get('origin') != 'external_camera_new_capture':
        raise ValueError('Only schema_version=1, origin=external_camera_new_capture permitted')
    if doc.get('source_split') != 'external_new_capture' or doc.get('contains_pklot_test') is not False:
        raise ValueError('PKLot train/valid/test and online TEST simulation are forbidden as new data')
    if doc.get('approved') is not True or not str(doc.get('approved_by', '')).strip():
        raise ValueError('Human approval and approved_by are required')
    if not str(doc.get('approved_at', '')).strip():
        raise ValueError('approved_at is required')
    rows = doc.get('samples')
    if not isinstance(rows, list) or len(rows) < minimum * 2 or len(rows) > 10000:
        raise ValueError('Insufficient or excessive approved crops')
    seen = set()
    counts = Counter()
    output = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or row.get('reviewed') is not True:
            raise ValueError(f'Crop {i}: explicit reviewed=true required')
        name = safe_relative(row.get('file'))
        if name.parts[0] != 'images' or len(name.parts) != 2:
            raise ValueError(f'Crop {i}: file must be images/<filename>')
        label = row.get('label')
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f'Crop {i}: label must be integer 0 or 1')
        frame = row.get('source_frame_id')
        slot = row.get('space_id')
        if not isinstance(frame, str) or not frame.strip() or not isinstance(slot, str) or not slot.strip():
            raise ValueError(f'Crop {i}: source_frame_id and space_id required')
        if any(x in frame.lower() for x in ('data/test', 'realtime_stream', 'pklot_test')):
            raise ValueError(f'Crop {i}: prohibited test-related source frame')
        key = (frame, slot)
        if key in seen:
            raise ValueError(f'Duplicate frame/space: {key}')
        seen.add(key)
        path = approved_dir / name
        if not path.is_file() or not path.resolve().is_relative_to(approved_dir.resolve()):
            raise FileNotFoundError(path)
        digest = sha256(path)
        if not re.fullmatch(r'[0-9a-f]{64}', str(row.get('sha256', ''))) or digest != row['sha256']:
            raise ValueError(f'Crop {i}: SHA256 mismatch')
        with Image.open(path) as img:
            img.verify()
        counts[label] += 1
        output.append({'file': f'train/{i:05d}.png', 'label': label, 'source_frame_id': frame,
                       'source_sha256': digest, 'source_file': str(path)})
    if min(counts[0], counts[1]) < minimum:
        raise ValueError(f'At least {minimum} reviewed samples/class required; found {dict(counts)}')
    return doc, output


def validation_samples(data_dir, max_per_class):
    from collections import defaultdict
    coco = json.loads((data_dir / '_annotations.coco.json').read_text(encoding='utf-8'))
    images = {im['id']: im for im in coco['images']}
    bins = defaultdict(list)
    for ann in coco['annotations']:
        cat = ann.get('category_id')
        if cat not in (1, 2) or ann.get('image_id') not in images:
            continue
        im = images[ann['image_id']]
        name = safe_relative(im['file_name'])
        if len(name.parts) != 1:
            raise ValueError('Nested validation images not supported')
        p = data_dir / name
        if not p.is_file():
            continue
        x, y, w, h = [float(v) for v in ann['bbox']]
        if w <= 5 or h <= 5:
            continue
        bins[cat - 1].append((p, (x, y, w, h), str(ann['id'])))
    rng = random.Random(73)
    rows = []
    for label in (0, 1):
        pool = sorted(bins[label], key=lambda r: r[2])
        rng.shuffle(pool)
        if len(pool) < max_per_class:
            raise ValueError(f'Not enough PKLot validation crops for class {label}')
        rows += [(label, *entry) for entry in pool[:max_per_class]]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--approved-dir', default='approved_online')
    ap.add_argument('--valid-dir', default='data/valid')
    ap.add_argument('--out', default='.stage7_dataset')
    ap.add_argument('--min-per-class', type=int, default=16)
    ap.add_argument('--valid-per-class', type=int, default=128)
    args = ap.parse_args()
    approved_dir = Path(args.approved_dir)
    doc, samples = approved_samples(approved_dir, args.min_per_class)
    val_samples = validation_samples(Path(args.valid_dir), args.valid_per_class)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    dataset_name = f"smart-parking-ct-r{os.environ['GITHUB_RUN_ID']}-a{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    archive_path = out / 'payload.zip'
    image_transform_note = 'PIL RGB crop from COCO bbox; resize 64x64 at model loader'
    manifest = {'stage': 'stage7-continuous-training', 'github_run_id': os.environ['GITHUB_RUN_ID'],
                'git_commit': os.environ['GITHUB_SHA'], 'dataset_name': dataset_name,
                'approved_dvc_sha256': sha256('approved_online.dvc'),
                'data_dvc_sha256': sha256('data.dvc'),
                'approved_by': doc['approved_by'], 'approved_at': doc['approved_at'],
                'approved_manifest_sha256': sha256(approved_dir / 'manifest.json'),
                'origin': 'external_camera_new_capture', 'train': [], 'valid': [],
                'checkpoints': {}, 'production_approved': False,
                'validation_source': 'data/valid', 'validation_note': image_transform_note}
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for row in samples:
            with Image.open(row['source_file']) as img:
                from io import BytesIO
                buffer = BytesIO()
                img.convert('RGB').save(buffer, format='PNG')
                raw = buffer.getvalue()
            z.writestr(row['file'], raw)
            manifest['train'].append({'file': row['file'], 'label': row['label'],
                                      'sha256': hashlib.sha256(raw).hexdigest()})
        for i, (label, src, (x, y, w, h), annid) in enumerate(val_samples):
            import math
            with Image.open(src) as im:
                im = im.convert('RGB')
                crop = im.crop((max(0, math.floor(x)), max(0, math.floor(y)),
                                min(im.width, math.ceil(x+w)), min(im.height, math.ceil(y+h))))
                if crop.width == 0 or crop.height == 0:
                    raise ValueError('Invalid validation bbox')
                from io import BytesIO
                buffer = BytesIO()
                crop.save(buffer, format='PNG')
                raw = buffer.getvalue()
            fname = f'valid/{i:05d}.png'
            z.writestr(fname, raw)
            manifest['valid'].append({'file': fname, 'label': label,
                                      'sha256': hashlib.sha256(raw).hexdigest()})
        for model, source in CHECKPOINTS.items():
            p = Path(source)
            if not p.is_file():
                raise FileNotFoundError(p)
            dest = f'checkpoint/{p.name}'
            z.write(p, dest)
            manifest['checkpoints'][model] = {'file': dest, 'sha256': sha256(p)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out / 'dataset-metadata.json').write_text(json.dumps({
        'title': dataset_name, 'id': 'nguynngthibo/' + dataset_name,
        'licenses': [{'name': 'unknown'}]}, indent=2), encoding='utf-8')
    with zipfile.ZipFile(archive_path) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(samples) + len(val_samples) + 3
    print('STAGE7_HUMAN_APPROVAL_VALIDATED_PASS')
    print('STAGE7_NO_PKLOT_TEST_TRAINING_PASS')
    print(f'STAGE7_PACK_PASS: {len(samples)} approved new crops; {len(val_samples)} PKLot valid crops')

if __name__ == '__main__':
    main()
