"""Local TEST_ONLY bridge: confirmed frame batch -> DVC -> GitHub push -> Actions.

Run once (for existing confirmed batch): python pipeline/stage7_auto_trigger.py --once
Run as watcher for later batches:     python pipeline/stage7_auto_trigger.py --watch
Use --dry-run to validate without changing DVC, Git, or network state.

This script runs locally with the user's existing Git/DVC credentials; the
Streamlit web process never receives a GitHub or DagsHub token.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / 'simulation_train'
CONF = ROOT / 'confirmed_train_sim'
BATCHES = ROOT / 'pipeline' / 'stage7_batches'


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verified_batch():
    manifest = json.loads((SIM / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('origin') != 'pklot_train_simulation' or
            manifest.get('source_split') != 'data/train' or
            manifest.get('test_only') is not True or
            manifest.get('production_approved') is not False):
        raise ValueError('Only TEST_ONLY PKLot train simulation is supported')
    frames = manifest['images']
    if len(frames) != 20 or len({str(r['image_id']) for r in frames}) != 20:
        raise ValueError('Expected 20 distinct simulated frames')
    by_id = {str(r['image_id']): r for r in frames}
    for row in frames:
        if row.get('source_split') != 'data/train':
            raise ValueError('Invalid source split')
        sim_name = Path(row['file_name']).name
        src_name = Path(row['source_file_name']).name
        if row['file_name'] != sim_name or row['source_file_name'] != src_name:
            raise ValueError('Unsafe source filename')
        original = ROOT / 'data' / 'train' / src_name
        source = SIM / 'images' / sim_name
        if not original.is_file() or not source.is_file():
            raise ValueError(f'Missing original or simulated image: {sim_name}')
        if sha256_file(original) != sha256_file(source):
            raise ValueError('Simulation frame differs from PKLot train original')
    label_files = sorted((CONF / 'frames').glob('*/labels.json'))
    if len(label_files) != 20:
        raise ValueError(f'Waiting for exactly 20 confirmed frames; have {len(label_files)}')
    seen_frames = set()
    counts = {'Empty': 0, 'Occupied': 0}
    digest = hashlib.sha256()
    digest.update((SIM / 'manifest.json').read_bytes())
    for file in label_files:
        record = json.loads(file.read_text(encoding='utf-8'))
        frame_id = str(record.get('source_image_id'))
        source = by_id.get(frame_id)
        if (not source or frame_id in seen_frames or
                record.get('source_image') != source['file_name'] or
                record.get('review_method') != 'manual_frame_review'):
            raise ValueError('Invalid or duplicated manual confirmation')
        seen_frames.add(frame_id)
        slots = record.get('slots', [])
        if not slots:
            raise ValueError('Empty confirmed frame')
        seen_slots = set()
        digest.update(file.relative_to(CONF).as_posix().encode())
        digest.update(file.read_bytes())
        for slot in slots:
            sid = int(slot['slot_id'])
            label = slot.get('confirmed_label')
            if sid in seen_slots or label not in counts:
                raise ValueError('Duplicated slot or unknown label')
            seen_slots.add(sid)
            if int(slot.get('label_id', -1)) != {'Empty': 0, 'Occupied': 1}[label]:
                raise ValueError('Inconsistent numeric label')
            name = slot['crop_file']
            if Path(name).name != name:
                raise ValueError('Unsafe crop path')
            crop = file.parent / name
            if not crop.is_file():
                raise ValueError(f'Missing crop: {crop}')
            from PIL import Image
            with Image.open(crop) as im:
                if im.size != (64, 64):
                    raise ValueError('Expected 64x64 confirmed crop')
                im.verify()
            counts[label] += 1
            digest.update(name.encode())
            digest.update(sha256_file(crop).encode())
    if len(seen_frames) != 20 or min(counts.values()) < 16:
        raise ValueError('Need 20 manually confirmed frames and >=16 examples per class')
    if sum(counts.values()) != 2000:
        raise ValueError(f'Expected 2000 confirmed crops, got {sum(counts.values())}')
    batch_id = digest.hexdigest()[:20]
    return {'batch_id': batch_id, 'origin': 'pklot_train_simulation',
            'mode': 'TEST_ONLY_NO_PROMOTION', 'frames': 20,
            'crops': sum(counts.values()), 'class_counts': counts,
            'data_version': 'confirmed_train_sim.dvc', 'production_approved': False}


def run(*args):
    print('+', ' '.join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def assert_safe_git():
    # No hidden stage/commit of unrelated code: all staged changes must be deliberate.
    result = subprocess.run(['git', 'diff', '--cached', '--name-only'], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise RuntimeError('Git index contains staged files. Commit/unstage them before auto trigger:\n' + result.stdout)
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    if branch != 'main':
        raise RuntimeError(f'Expected local branch main, got {branch!r}')
    remote = subprocess.run(['git', 'remote', 'get-url', 'github'], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    if 'thaibao3214/smart-parking-mlops' not in remote:
        raise RuntimeError('Refusing to push to unexpected github remote: ' + remote)


def dispatch(batch, dry_run):
    path = BATCHES / f"{batch['batch_id']}.json"
    if path.exists():
        print('STAGE7_BATCH_ALREADY_SUBMITTED:', path.name, flush=True)
        return False
    print('STAGE7_BATCH_VERIFIED:', json.dumps(batch, ensure_ascii=False), flush=True)
    if dry_run:
        print('STAGE7_DRY_RUN_NO_CHANGES', flush=True)
        return True
    assert_safe_git()
    # DVC content uploaded before Git pointer/marker is made visible on GitHub.
    run('dvc', 'add', 'simulation_train')
    run('dvc', 'add', 'confirmed_train_sim')
    run('dvc', 'push', 'simulation_train.dvc', 'confirmed_train_sim.dvc', '-r', 'origin')
    BATCHES.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(batch, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    run('git', 'add', '--', 'simulation_train.dvc', 'confirmed_train_sim.dvc',
        path.relative_to(ROOT).as_posix())
    run('git', 'commit', '-m', f"Stage7 TEST_ONLY confirmed batch {batch['batch_id']}")
    run('git', 'push', 'github', 'HEAD:main')
    print('STAGE7_AUTO_TRIGGER_PUSH_PASS:', batch['batch_id'], flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--once', action='store_true')
    choice.add_argument('--watch', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=15)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error('poll-seconds must be >=5')
    while True:
        try:
            batch = verified_batch()
            changed = dispatch(batch, args.dry_run)
            if changed and not args.dry_run:
                print('GitHub Actions push trigger has been published', flush=True)
        except (FileNotFoundError, ValueError) as error:
            print('STAGE7_WAITING_FOR_CONFIRMED_BATCH:', error, flush=True)
            if args.once:
                sys.exit(1)
        except (subprocess.CalledProcessError, RuntimeError) as error:
            print('STAGE7_AUTO_TRIGGER_ERROR:', error, flush=True)
            sys.exit(1)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
