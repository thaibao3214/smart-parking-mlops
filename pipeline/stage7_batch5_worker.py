"""Batch 005 automated PKLot TRAIN TEST_ONLY simulation and DVC submission.

Runs in a detached local process, launched by password-protected Streamlit UI.
Never uses model predictions as training labels; labels/crops come from COCO TRAIN.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'pipeline'))
sys.path.insert(0, str(ROOT / 'dashboard'))

from stage7_gt_labels import GT_METHOD, LABELS, coordinates, expected_annotations, validate_gt_record

BATCH = 'batch_005'
STATUS_DIR = Path.home() / '.smartparking_stage7_control'
STATUS_FILE = STATUS_DIR / 'auto_batch_005.json'
LOCK_FILE = STATUS_DIR / 'auto_batch_005.lock'
GLOBAL_LOCK = STATUS_DIR / 'auto_batch_active.lock'

def configure_batch(batch):
    global BATCH, STATUS_FILE, LOCK_FILE
    if not re.fullmatch(r'batch_[0-9]{3,6}', batch):
        raise ValueError('Unsafe batch name')
    BATCH = batch
    STATUS_FILE = STATUS_DIR / f'auto_{batch}.json'
    LOCK_FILE = STATUS_DIR / f'auto_{batch}.lock'


def utc():
    return datetime.now(timezone.utc).isoformat()


def status(state, **values):
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    old = {}
    if STATUS_FILE.exists():
        try:
            old = json.loads(STATUS_FILE.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
    old.update(values)
    old.update(state=state, updated_at=utc(), batch=BATCH, mode='TEST_ONLY_NO_PROMOTION')
    temp = STATUS_DIR / (f'.auto_{BATCH}_' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(STATUS_FILE)


def assert_local_api(url):
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        raise ValueError('FastAPI must be a local HTTP host (127.0.0.1 or localhost)')
    return url.rstrip('/')


def load_manifest():
    sim = ROOT / 'simulation_batches' / BATCH
    doc = json.loads((sim / 'manifest.json').read_text(encoding='utf-8'))
    frames = doc.get('images')
    if (doc.get('batch_name') != BATCH or doc.get('source_split') != 'data/train'
            or doc.get('origin') != 'pklot_train_simulation'
            or doc.get('test_only') is not True
            or doc.get('production_approved') is not False
            or not isinstance(frames, list) or len(frames) != 20
            or len({str(f['image_id']) for f in frames}) != 20
            or len({str(f['source_file_name']) for f in frames}) != 20):
        raise ValueError('Invalid immutable batch_005 manifest')
    for f in frames:
        if (f.get('source_split') != 'data/train' or
                any(Path(f.get(k, '')).name != f.get(k) or not f.get(k)
                    for k in ('file_name', 'source_file_name'))):
            raise ValueError('Unsafe/incorrect image name or source split')
    return sim, frames


def make_if_needed():
    sim = ROOT / 'simulation_batches' / BATCH
    if sim.exists():
        # Only resume an auto-owned session, never take over a manually made batch.
        previous = json.loads(STATUS_FILE.read_text(encoding='utf-8')) if STATUS_FILE.exists() else {}
        if not previous.get('created_by_auto_worker'):
            raise RuntimeError('batch_005 already exists from another source; refusing overwrite')
    else:
        from prepare_train_simulation import make_batch
        last_error = None
        for cell in (32, 48, 64):
            try:
                make_batch(BATCH, 20, 20261126 + int(BATCH.split('_')[1]) - 5, cell)
                status('created', created_by_auto_worker=True, cell=cell, completed_frames=0, message='Created 20 new PKLot TRAIN frames')
                break
            except RuntimeError as exc:
                if 'No 100-space layout group' not in str(exc):
                    raise
                last_error = exc
        if not sim.exists():
            raise RuntimeError(str(last_error or 'Cannot choose 20 suitable frames'))
    return load_manifest()


def generate_labels(sim, frame, record):
    from PIL import Image
    conf = ROOT / 'confirmed_batches' / BATCH
    folder = conf / 'frames' / f"{frame['image_id']}_{Path(frame['file_name']).stem}"
    label_path = folder / 'labels.json'
    if label_path.exists():
        payload = json.loads(label_path.read_text(encoding='utf-8'))
        validate_gt_record(ROOT, sim, conf, frame, payload, label_path)
        return 100
    if folder.exists():
        raise RuntimeError('Incomplete existing GT frame folder; no overwrite: ' + str(folder))
    folder.parent.mkdir(parents=True, exist_ok=True)
    temp = folder.parent / ('.tmp_gt_' + uuid.uuid4().hex)
    temp.mkdir()
    try:
        anns = expected_annotations(ROOT, frame)
        slots = []
        with Image.open(sim / 'images' / frame['file_name']) as im:
            img = im.convert('RGB')
            for index, ann in enumerate(anns, 1):
                label, label_id = LABELS[int(ann['category_id'])]
                x, y, w, h = coordinates(ann['bbox'], img.size)
                crop_name = f'slot_{index:03d}.png'
                img.crop((x, y, x+w, y+h)).resize((64, 64)).save(temp / crop_name)
                slots.append(dict(slot_id=index, crop_file=crop_name, bbox_xywh=[x, y, w, h],
                                  source_annotation_id=int(ann['id']), predicted_label='Unknown',
                                  confirmed_label=label, label_id=label_id))
        payload = dict(source_image=frame['file_name'], source_file_name=frame['source_file_name'],
                       source_image_id=frame['image_id'], source_split='data/train',
                       origin='pklot_train_simulation', test_only=True,
                       production_approved=False, model_type=record['model_type'],
                       model_version=record.get('model_version'), model_source=record.get('model_source'),
                       review_method=GT_METHOD, verified_at_utc=utc(), slots=slots)
        (temp / 'labels.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        validate_gt_record(ROOT, sim, conf, frame, payload, temp / 'labels.json')
        temp.rename(folder)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return len(slots)


def run(api_url):
    from online_stream import _predict, _record_path
    from stage7_auto_trigger import verify, dispatch, verify_code_ready
    from hashlib import sha256

    api_url = assert_local_api(api_url)
    status('preflight', message='Checking Stage 7 version and Git index', completed_frames=0)
    verify_code_ready()
    # The existing trigger must not commit or push staged unrelated files.
    import subprocess
    staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True)
    if staged.strip():
        raise RuntimeError('Git index is not empty. Commit/stash staged files first: ' + staged[:500])
    marker_dir = ROOT / 'pipeline' / 'stage7_batches'
    if any((m.get('batch_name') == BATCH for p in marker_dir.glob('*.json')
            for m in [json.loads(p.read_text(encoding='utf-8'))])):
        raise RuntimeError('Batch 005 already submitted. Refusing repeat DVC/Git push')
    status('creating', message='Selecting 20 unused COCO TRAIN frames')
    sim, frames = make_if_needed()
    conf = ROOT / 'confirmed_batches' / BATCH
    for i, frame in enumerate(frames):
        if i:
            time.sleep(5)  # camera-style cadence; no long wait in Streamlit UI
        src = sim / 'images' / frame['file_name']
        original = ROOT / 'data' / 'train' / frame['source_file_name']
        if not src.is_file() or not original.is_file() or sha256(src.read_bytes()).digest() != sha256(original.read_bytes()).digest():
            raise RuntimeError('Frame is not an exact copy of PKLot TRAIN image')
        pred_path = _record_path(sim, frame, 'mobilenetv3')
        if pred_path.exists():
            record = json.loads(pred_path.read_text(encoding='utf-8'))
        else:
            record = _predict(sim, frame, api_url, 'mobilenetv3', 'pklot_train_simulation', 'data/train')
            preds = record.get('predictions') or []
            if (len(preds) != 100 or len({int(x['id']) for x in preds}) != 100
                    or any(str(x['status']).lower() not in ('empty', 'occupied') for x in preds)):
                raise RuntimeError(f'FastAPI returned invalid 100-slot predictions in frame {i+1}')
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = pred_path.with_name('.' + pred_path.name + '.' + uuid.uuid4().hex + '.tmp')
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(pred_path)
        if (str(record.get('image_id')) != str(frame['image_id'])
                or record.get('file_name') != frame['file_name']
                or record.get('source_split') != 'data/train'
                or record.get('model_type') != 'mobilenetv3'):
            raise RuntimeError('Existing prediction does not match Batch 005 frame/model')
        crops = generate_labels(sim, frame, record)
        status('simulating', completed_frames=i+1, total_frames=20,
               completed_crops=(i+1)*crops,
               message=f'Frame {i+1}/20: FastAPI inference + COCO TRAIN ground truth validated')
    status('verifying', message='Running original Stage 7 immutable batch validation')
    batch = verify(BATCH)
    if batch.get('mode') != 'TEST_ONLY_NO_PROMOTION' or batch.get('crops') != 2000 or batch.get('frames') != 20:
        raise RuntimeError('Unexpected Stage 7 verification metrics')
    status('verified', batch_id=batch['batch_id'], completed_frames=20, completed_crops=2000,
           message='Dry Run PASS; preparing DVC push')
    # Existing dispatch performs DVC add/push + marker-only Git commit/push.
    # GitHub push event triggers the real CI workflow; no production promotion.
    status('publishing', message='Uploading immutable TEST_ONLY batch to DVC and GitHub')
    dispatch(batch, dry_run=False)
    status('submitted', message='DVC & Git push completed; GitHub Actions will handle Kaggle/MLflow/Evaluation Gate',
           batch_id=batch['batch_id'], completed_frames=20, completed_crops=2000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--api-url', default='http://127.0.0.1:8000')
    p.add_argument('--batch', default='batch_005')
    args = p.parse_args()
    configure_batch(args.batch)
    try:
        run(args.api_url)
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}', message='Stopped safely, no Production promotion')
        traceback.print_exc()
        sys.exit(1)
    finally:
        LOCK_FILE.unlink(missing_ok=True)
        # Only release a shared lock owned by this batch.
        if GLOBAL_LOCK.is_file():
            try:
                if GLOBAL_LOCK.read_text(encoding='utf-8').strip() == BATCH:
                    GLOBAL_LOCK.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == '__main__':
    main()
