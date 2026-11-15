"""Immutable-batch TEST_ONLY local watcher; no secrets in Streamlit.

Supports --batch batch_002 for new independent simulation sessions and
--legacy for the original completed session. A commit changes only one
batch marker, the two DVC pointer files and nothing else.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from stage7_exclude_frame import load_exclusions

ROOT = Path(__file__).resolve().parent.parent
BATCHES = ROOT / 'pipeline' / 'stage7_batches'


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def paths_for(batch):
    if batch == 'legacy':
        return ROOT / 'simulation_train', ROOT / 'confirmed_train_sim', 'simulation_train.dvc', 'confirmed_train_sim.dvc'
    if not re.fullmatch(r'batch_[0-9]{3,6}', batch):
        raise ValueError('Expected batch_002, batch_003, ...')
    return (ROOT / 'simulation_batches' / batch, ROOT / 'confirmed_batches' / batch,
            'simulation_batches.dvc', 'confirmed_batches.dvc')


def verify(batch):
    from PIL import Image
    sim, conf, sim_pointer, conf_pointer = paths_for(batch)
    manifest_path = sim / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('origin') != 'pklot_train_simulation' or
        manifest.get('source_split') != 'data/train' or
        manifest.get('test_only') is not True or
        manifest.get('production_approved') is not False):
        raise ValueError('Not PKLot TRAIN TEST_ONLY')
    if batch != 'legacy' and manifest.get('batch_name') != batch:
        raise ValueError('Manifest batch mismatch')
    frames = manifest['images']
    if len(frames) != 20 or len({str(f['image_id']) for f in frames}) != 20:
        raise ValueError('Need exactly 20 distinct source frames')
    mapped = {str(f['image_id']): f for f in frames}
    used = set()
    for entry in frames:
        source_name = Path(entry['source_file_name']).name
        image_name = Path(entry['file_name']).name
        if (source_name != entry['source_file_name'] or image_name != entry['file_name'] or
            entry.get('source_split') != 'data/train'):
            raise ValueError('Unsafe filename/source split')
        src = ROOT / 'data' / 'train' / source_name
        dest = sim / 'images' / image_name
        if not src.is_file() or not dest.is_file() or checksum(src) != checksum(dest):
            raise ValueError(f'Frame mismatch: {image_name}')
        used.add(source_name)
    if batch != 'legacy':
        previous = [ROOT / 'simulation_train' / 'manifest.json']
        previous += list((ROOT / 'simulation_batches').glob('*/manifest.json'))
        for old in previous:
            if old == manifest_path or not old.exists():
                continue
            previous_doc = json.loads(old.read_text(encoding='utf-8'))
            other = {str(f.get('source_file_name', '')) for f in previous_doc.get('images', [])}
            if used & other:
                raise ValueError('New batch reuses previously simulated frame')
    excluded_ids = load_exclusions(sim, conf, manifest)
    required_count = 20 - len(excluded_ids)
    labels = sorted((conf / 'frames').glob('*/labels.json'))
    if len(labels) < required_count or len(labels) > 20:
        raise ValueError(f'Waiting for {required_count} reviewed frames: got {len(labels)}')
    counts = {'Empty': 0, 'Occupied': 0}
    seen = set()
    digest = hashlib.sha256(manifest_path.read_bytes())
    if excluded_ids:
        digest.update((conf / 'excluded_frames.json').read_bytes())
    for path in labels:
        record = json.loads(path.read_text(encoding='utf-8'))
        frame_id = str(record.get('source_image_id'))
        ref = mapped.get(frame_id)
        if frame_id in excluded_ids:
            continue  # Keep original on disk, but never train on this frame.
        if (not ref or frame_id in seen or record.get('source_image') != ref['file_name'] or
            record.get('review_method') != 'manual_frame_review'):
            raise ValueError('Invalid manual confirmation / wrong frame')
        seen.add(frame_id)
        digest.update(path.relative_to(conf).as_posix().encode())
        digest.update(path.read_bytes())
        slots = record.get('slots', [])
        if not slots:
            raise ValueError('Empty confirmation')
        seen_slots = set()
        for slot in slots:
            sid = int(slot['slot_id'])
            label = slot['confirmed_label']
            if sid in seen_slots or label not in counts or int(slot.get('label_id', -1)) != {'Empty': 0, 'Occupied': 1}[label]:
                raise ValueError('Duplicate slot / invalid label')
            seen_slots.add(sid)
            name = slot['crop_file']
            if Path(name).name != name:
                raise ValueError('Unsafe crop path')
            crop = path.parent / name
            with Image.open(crop) as im:
                if im.size != (64, 64):
                    raise ValueError('Expected 64x64 crop')
                im.verify()
            counts[label] += 1
            digest.update(name.encode())
            digest.update(checksum(crop).encode())
    if len(seen) != required_count or min(counts.values()) < 16:
        raise ValueError(f'Expected {required_count} valid reviewed frames across both classes; got {counts}')
    if sum(counts.values()) != required_count * 100:
        raise ValueError(f'Expected {required_count * 100} crops after exclusions: {counts}')
    return {'batch_id': digest.hexdigest()[:20], 'batch_name': batch,
            'origin': 'pklot_train_simulation', 'mode': 'TEST_ONLY_NO_PROMOTION',
            'frames': required_count, 'crops': sum(counts.values()), 'class_counts': counts,
            'excluded_source_image_ids': sorted(excluded_ids),
            'sim_pointer': sim_pointer, 'confirmed_pointer': conf_pointer,
            'production_approved': False}


def run(*args):
    print('+', ' '.join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def dispatch(batch, dry_run):
    marker = BATCHES / (batch['batch_id'] + '.json')
    if marker.exists():
        print('STAGE7_BATCH_ALREADY_SUBMITTED:', marker.name)
        return
    print('STAGE7_BATCH_VERIFIED:', json.dumps(batch, ensure_ascii=False))
    if dry_run:
        print('STAGE7_DRY_RUN_NO_CHANGES')
        return
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True)
    if staged.strip():
        raise RuntimeError('Git index not empty; commit or unstage unrelated changes first:\n' + staged)
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(['git', 'remote', 'get-url', 'github'], cwd=ROOT, text=True).strip()
    if branch != 'main' or 'thaibao3214/smart-parking-mlops' not in remote:
        raise RuntimeError('Unsafe branch or remote')
    sim, conf, sim_ptr, conf_ptr = paths_for(batch['batch_name'])
    # Upload data BEFORE publishing a batch marker in Git.
    run('dvc', 'add', str(sim.relative_to(ROOT)), str(conf.relative_to(ROOT)))
    run('dvc', 'push', sim_ptr, conf_ptr, '-r', 'origin')
    batch['sim_dvc_sha256'] = checksum(ROOT / sim_ptr)
    batch['online_dvc_sha256'] = checksum(ROOT / conf_ptr)
    BATCHES.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(batch, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    run('git', 'add', '--', sim_ptr, conf_ptr, marker.relative_to(ROOT).as_posix())
    run('git', 'commit', '-m', f"Stage7 TEST_ONLY confirmed batch {batch['batch_id']}")
    run('git', 'push', 'github', 'HEAD:main')
    print('STAGE7_AUTO_TRIGGER_PUSH_PASS:', batch['batch_id'])


def main():
    p = argparse.ArgumentParser()
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--once', action='store_true')
    choice.add_argument('--watch', action='store_true')
    p.add_argument('--batch', default='batch_002')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--poll-seconds', type=int, default=15)
    a = p.parse_args()
    if a.poll_seconds < 5:
        p.error('--poll-seconds must be >= 5')
    while True:
        try:
            dispatch(verify(a.batch), a.dry_run)
        except (OSError, ValueError, KeyError, FileNotFoundError) as err:
            print('STAGE7_WAITING_FOR_CONFIRMED_BATCH:', err)
            if a.once:
                sys.exit(1)
        except (subprocess.CalledProcessError, RuntimeError) as err:
            print('STAGE7_AUTO_TRIGGER_ERROR:', err)
            sys.exit(1)
        if a.once:
            return
        time.sleep(a.poll_seconds)

if __name__ == '__main__':
    main()
