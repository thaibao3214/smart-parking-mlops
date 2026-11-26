"""Package human-confirmed PKLot TRAIN simulation for Stage 7 TEST_ONLY.
Never misrepresent simulated PKLot frames as newly captured external camera data.
"""
import hashlib
import json
import os
import zipfile
from pathlib import Path
from PIL import Image
from stage7_prepare import CHECKPOINTS, sha256, validation_samples
from stage7_exclude_frame import load_exclusions

SIM = Path(os.environ.get('STAGE7_SIM_DIR', 'simulation_train'))
CONFIRMED = Path(os.environ.get('STAGE7_CONFIRMED_DIR', 'confirmed_train_sim'))
SIM_POINTER = os.environ.get('STAGE7_SIM_POINTER', 'simulation_train.dvc')
CONF_POINTER = os.environ.get('STAGE7_CONFIRMED_POINTER', 'confirmed_train_sim.dvc')
BATCH_ID = os.environ.get('STAGE7_BATCH_ID', '')



def confirmed_rows():
    sm = json.loads((SIM / 'manifest.json').read_text(encoding='utf-8'))
    if (sm.get('origin') != 'pklot_train_simulation' or
            sm.get('source_split') != 'data/train' or
            sm.get('test_only') is not True or sm.get('production_approved') is not False):
        raise ValueError('Simulation manifest must clearly identify PKLot TRAIN TEST_ONLY')
    if BATCH_ID and sm.get('batch_name') != Path(SIM).name:
        raise ValueError('Batch name mismatch')
    mapping = {str(f['image_id']): f for f in sm['images']}
    if len(mapping) != len(sm['images']) or len(mapping) != 20:
        raise ValueError('Expected 20 distinct PKLot train simulation frames')
    for f in mapping.values():
        if f.get('source_split') != 'data/train':
            raise ValueError('Not sourced from PKLot train')
        original = Path('data/train') / Path(f['source_file_name']).name
        simulated = SIM / 'images' / Path(f['file_name']).name
        if not original.is_file() or not simulated.is_file() or sha256(original) != sha256(simulated):
            raise ValueError('Simulated frame does not match original PKLot TRAIN frame')
    excluded_ids = load_exclusions(SIM, CONFIRMED, sm)
    required_count = 20 - len(excluded_ids)
    files = sorted((CONFIRMED / 'frames').glob('*/labels.json'))
    if len(files) < required_count or len(files) > 20:
        raise ValueError(f'Expected {required_count} reviewed frames, received {len(files)}')
    rows = []
    used_images = set()
    counts = {0: 0, 1: 0}
    for labels in files:
        payload = json.loads(labels.read_text(encoding='utf-8'))
        image_id = str(payload['source_image_id'])
        if image_id in excluded_ids:
            continue  # Deliberately rejected source, never package its crops.
        ref = mapping.get(image_id)
        if ref is None or ref['file_name'] != payload['source_image'] or image_id in used_images:
            raise ValueError('Unexpected, duplicate, or wrong-source confirmation')
        method = payload.get('review_method')
        if method == 'pklot_coco_ground_truth':
            from stage7_gt_labels import validate_gt_record
            validate_gt_record(Path.cwd(), SIM, CONFIRMED, ref, payload, labels)
        elif method != 'manual_frame_review':
            raise ValueError('Invalid frame review provenance')
        used_images.add(image_id)
        slots = payload.get('slots', [])
        if not slots:
            raise ValueError(f'No slots for {image_id}')
        seen_slots = set()
        for slot in slots:
            slot_id = int(slot['slot_id'])
            if slot_id in seen_slots:
                raise ValueError('Duplicate slot id')
            seen_slots.add(slot_id)
            label = slot['confirmed_label']
            index = {'Empty': 0, 'Occupied': 1}.get(label)
            if index is None or int(slot['label_id']) != index:
                raise ValueError('Invalid or inconsistent confirmed label')
            name = Path(slot['crop_file']).name
            if name != slot['crop_file']:
                raise ValueError('Unsafe crop filename')
            src = labels.parent / name
            if not src.is_file():
                raise FileNotFoundError(src)
            with Image.open(src) as im:
                im.verify()
            counts[index] += 1
            rows.append((index, src, ref['source_file_name'], image_id, slot_id))
    if len(used_images) != required_count or min(counts.values()) < 16:
        raise ValueError(f'Incomplete reviewed batch or missing class: {counts}')
    if len(rows) != required_count * 100:
        raise ValueError(f'Expected exactly {required_count * 100} crops, got {len(rows)}')
    return rows



def dvc_checksum_matches(expected, path):
    raw = Path(path).read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    crlf = lf.replace(b'\n', b'\r\n')
    return expected in {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(lf).hexdigest(),
        hashlib.sha256(crlf).hexdigest(),
    }

def main():
    if not os.environ.get('GITHUB_RUN_ID') or not os.environ.get('GITHUB_SHA'):
        raise ValueError('Only package from GitHub Actions')
    dest = Path('.stage7_test_dataset')
    if dest.exists():
        raise RuntimeError('Output exists; use a clean Actions checkout')
    if BATCH_ID:
        batch_marker = Path('pipeline/stage7_batches') / (BATCH_ID + '.json')
        marker = json.loads(batch_marker.read_text(encoding='utf-8'))
        if (marker.get('batch_id') != BATCH_ID or marker.get('batch_name') != SIM.name or
                marker.get('sim_pointer') != SIM_POINTER or marker.get('confirmed_pointer') != CONF_POINTER or
                not dvc_checksum_matches(marker.get('sim_dvc_sha256'), SIM_POINTER) or
                not dvc_checksum_matches(marker.get('online_dvc_sha256'), CONF_POINTER) or
                marker.get('production_approved') is not False):
            raise ValueError('Source batch marker/DVC pointer mismatch: fail closed')
        current_manifest = json.loads((SIM / 'manifest.json').read_text(encoding='utf-8'))
        current_excluded = sorted(load_exclusions(SIM, CONFIRMED, current_manifest))
        if (marker.get('excluded_source_image_ids', []) != current_excluded or
                marker.get('frames') != 20 - len(current_excluded) or
                marker.get('crops') != 100 * (20 - len(current_excluded))):
            raise ValueError('Batch marker exclusion/data counts mismatch')
    rows_train = confirmed_rows()
    sim_manifest = json.loads((SIM / 'manifest.json').read_text(encoding='utf-8'))
    excluded_ids = load_exclusions(SIM, CONFIRMED, sim_manifest)
    rows_valid = validation_samples(Path('data/valid'), 128)
    train_frames = {r[2] for r in rows_train}
    valid_frames = {src.name for _, src, _, _ in rows_valid}
    if train_frames & valid_frames:
        raise ValueError('Overlapping source filenames in train and validation')
    dataset_name = f"smart-parking-stage7-test-r{os.environ['GITHUB_RUN_ID']}-a{os.environ.get('GITHUB_RUN_ATTEMPT','1')}"
    manifest = {
        'stage': 'stage7-test-only', 'origin': 'pklot_train_test_only',
        'train_source': 'Manually confirmed frames simulated from PKLot data/train',
        'reviewed_frame_count': 20 - len(excluded_ids),
        'excluded_source_image_ids': sorted(excluded_ids),
        'validation_source': 'PKLot data/valid (technical holdout, NOT blind independent)',
        'test_only': True, 'production_approved': False,
        'github_run_id': os.environ['GITHUB_RUN_ID'], 'git_commit': os.environ['GITHUB_SHA'],
        'dataset_name': dataset_name, 'approved_dvc_sha256': 'NOT_APPLICABLE_TEST_ONLY',
        'data_dvc_sha256': sha256('data.dvc'),
        'online_dvc_sha256': sha256(CONF_POINTER),
        'sim_dvc_sha256': sha256(SIM_POINTER),
        'online_dvc_pointer': CONF_POINTER, 'sim_dvc_pointer': SIM_POINTER,
        'batch_id': BATCH_ID, 'batch_name': SIM.name,
        'train': [], 'valid': [], 'checkpoints': {},
        'warning': 'Technical pipeline validation only; PKLot TRAIN simulation, NOT new camera captures',
    }
    dest.mkdir()
    with zipfile.ZipFile(dest / 'payload.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for i, (label, src, original, image_id, slot_id) in enumerate(rows_train):
            raw = src.read_bytes()
            path = f'train/{i:05d}.png'
            archive.writestr(path, raw)
            manifest['train'].append({
                'file': path, 'label': label,
                'sha256': hashlib.sha256(raw).hexdigest(),
                'source_split': 'data/train', 'source_frame': original,
                'source_image_id': image_id, 'slot_id': slot_id,
                'review_method': 'manual_frame_review',
            })
        for i, (label, src, bbox, ann_id) in enumerate(rows_valid):
            # Valid annotations are evaluation ground truth, not online confirmations.
            import math
            from io import BytesIO
            x, y, w, h = bbox
            with Image.open(src) as im:
                im = im.convert('RGB')
                crop = im.crop((max(0, math.floor(x)), max(0, math.floor(y)),
                                min(im.width, math.ceil(x+w)), min(im.height, math.ceil(y+h))))
                buf = BytesIO()
                crop.save(buf, format='PNG')
                raw = buf.getvalue()
            path = f'valid/{i:05d}.png'
            archive.writestr(path, raw)
            manifest['valid'].append({
                'file': path, 'label': label,
                'sha256': hashlib.sha256(raw).hexdigest(),
                'source_split': 'data/valid', 'source_frame': src.name, 'annotation_id': ann_id,
            })
        for model, cp in CHECKPOINTS.items():
            path = Path(cp)
            if not path.is_file():
                raise FileNotFoundError(cp)
            stored = f'checkpoint/{path.name}'
            archive.write(path, stored)
            manifest['checkpoints'][model] = {'file': stored, 'sha256': sha256(path)}
    with zipfile.ZipFile(dest / 'payload.zip') as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(rows_train) + len(rows_valid) + 3:
            raise ValueError('Corrupt or incomplete payload')
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (dest / 'dataset-metadata.json').write_text(json.dumps({
        'title': dataset_name, 'id': f'nguynngthibo/{dataset_name}',
        'licenses': [{'name': 'unknown'}],
    }, indent=2), encoding='utf-8')
    print(f'STAGE7_SIM_CONFIRMED_PREP_PASS train={len(rows_train)} valid={len(rows_valid)}')
    print('STAGE7_TEST_ONLY_PRODUCTION_BLOCKED')


if __name__ == '__main__':
    main()
