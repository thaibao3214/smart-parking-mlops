"""Read-only preflight for Smart Parking MLOps. Run at project root.
Does not import application code, write reports, or change files.
"""
import argparse
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = 20
results = []


def check(section, name, state, detail):
    results.append((section, name, state, str(detail)))
    print(f"  [{state:5}] {name}: {detail}")


def run(*cmd, timeout=45):
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=timeout,
                           stdin=subprocess.DEVNULL)
        return p.returncode, (p.stdout + '\n' + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, str(exc)


def section(title):
    print(f"\n=== {title} ===")


def exists(rel, label, missing='FAIL'):
    value = (ROOT / rel).is_file()
    check('FILES', label, 'PASS' if value else missing, rel)
    return value


def parse_python(rel):
    p = ROOT / rel
    if not p.is_file():
        check('CODE', rel, 'FAIL', 'File missing')
        return
    try:
        ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
        check('CODE', rel, 'PASS', 'Python syntax OK')
    except (SyntaxError, UnicodeError) as exc:
        check('CODE', rel, 'FAIL', str(exc))


def inspect_data():
    section('ONLINE DATA: 20 FRAMES / CONFIRMED CROPS')
    stream = ROOT / 'data/realtime_stream'
    manifest = stream / 'manifest.json'
    coco = stream / '_annotations.coco.json'
    frame_root = ROOT / 'data/confirmed_online/frames'

    if manifest.is_file():
        try:
            m = json.loads(manifest.read_text(encoding='utf-8'))
            items = m.get('images', [])
            if not isinstance(items, list):
                raise ValueError('manifest.images must be a list')
            names = [Path(x['file_name']).name for x in items]
            missing = [n for n in names if not (stream / 'images' / n).is_file()]
            valid = len(items) == EXPECTED and len(set(names)) == EXPECTED and not missing
            check('DATA', 'manifest and images', 'PASS' if valid else 'FAIL',
                  f'{len(items)} listed; {len(set(names))} unique; {len(missing)} missing')
        except (OSError, KeyError, ValueError, TypeError) as exc:
            check('DATA', 'manifest and images', 'FAIL', str(exc))
    else:
        check('DATA', 'manifest and images', 'FAIL', 'Missing manifest.json')

    if coco.is_file():
        try:
            d = json.loads(coco.read_text(encoding='utf-8'))
            images = d.get('images', [])
            annotations = d.get('annotations', [])
            ids = {i['id'] for i in images}
            orphan = sum(a.get('image_id') not in ids for a in annotations)
            ok = len(images) == EXPECTED and not orphan
            check('DATA', 'online COCO', 'PASS' if ok else 'FAIL',
                  f'{len(images)} images, {len(annotations)} boxes, {orphan} orphan annotations')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            check('DATA', 'online COCO', 'FAIL', str(exc))
    else:
        check('DATA', 'online COCO', 'FAIL', 'Missing COCO JSON')

    files = sorted(frame_root.glob('*/labels.json')) if frame_root.is_dir() else []
    slot_count = 0
    crops = 0
    invalid = []
    labels = Counter()
    corrections = 0
    sources = []
    versions = Counter()
    try:
        from PIL import Image
    except ImportError:
        Image = None
        check('DATA', 'Pillow', 'WARN', 'Not installed; cannot verify image content')

    for labels_file in files:
        try:
            d = json.loads(labels_file.read_text(encoding='utf-8'))
            sources.append(d.get('source_image'))
            versions[str(d.get('model_version'))] += 1
            slots = d.get('slots')
            if not isinstance(slots, list) or not slots:
                invalid.append(f'{labels_file}: empty slots')
                continue
            seen_slots = set()
            for s in slots:
                slot_count += 1
                sid = s.get('slot_id')
                if sid in seen_slots:
                    invalid.append(f'{labels_file}: duplicate slot_id {sid}')
                seen_slots.add(sid)
                label = s.get('confirmed_label')
                if label not in ('Empty', 'Occupied'):
                    invalid.append(f'{labels_file}: invalid label {label}')
                else:
                    labels[label] += 1
                    if s.get('label_id') != (0 if label == 'Empty' else 1):
                        invalid.append(f'{labels_file}: mismatched label_id {sid}')
                if s.get('predicted_label') != label:
                    corrections += 1
                filename = s.get('crop_file')
                if not isinstance(filename, str) or Path(filename).name != filename or not filename.lower().endswith('.png'):
                    invalid.append(f'{labels_file}: unsafe/invalid crop filename {filename}')
                    continue
                crop = labels_file.parent / filename
                if not crop.is_file():
                    invalid.append(f'{labels_file}: missing {filename}')
                    continue
                crops += 1
                if Image is not None:
                    try:
                        with Image.open(crop) as img:
                            if img.size != (64, 64) or img.mode != 'RGB':
                                invalid.append(f'{crop}: size={img.size}, mode={img.mode}')
                            img.verify()
                    except (OSError, ValueError) as exc:
                        invalid.append(f'{crop}: {exc}')
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            invalid.append(f'{labels_file}: {exc}')

    count_ok = len(files) == EXPECTED and slot_count == 2000 and crops == 2000
    check('DATA', 'confirmed dataset counts', 'PASS' if count_ok else 'FAIL',
          f'{len(files)} frames, {slot_count} labels, {crops} crop files')
    check('DATA', 'label distribution', 'PASS' if labels['Empty'] and labels['Occupied'] else 'FAIL',
          f'Empty={labels["Empty"]}, Occupied={labels["Occupied"]}')
    check('DATA', 'crop and label integrity', 'PASS' if not invalid else 'FAIL',
          f'{len(invalid)} errors; ' + ('; '.join(invalid[:3]) if invalid else 'all valid'))
    check('DATA', 'review audit', 'WARN' if corrections == 0 else 'PASS',
          f'{corrections} manual corrections; zero does NOT prove labels are correct')
    check('DATA', 'source uniqueness', 'PASS' if len(sources) == len(set(sources)) else 'FAIL',
          f'{len(set(sources))} distinct source images; model versions={dict(versions)}')
    if (ROOT / 'data/test/_annotations.coco.json').is_file():
        check('DATA', 'test-set isolation', 'WARN',
              'Online frames came from PKLot test; exclude their source images from v2 evaluation')
    else:
        check('DATA', 'test-set isolation', 'WARN', 'Test COCO not present locally')


def inspect_git_dvc(check_remote):
    section('GIT / GITHUB / DAGSHUB / DVC')
    for cmd, title in [
        (('git', 'status', '--short'), 'Git working tree'),
        (('git', 'log', '-1', '--oneline'), 'HEAD commit'),
        (('git', 'remote', '-v'), 'Git remotes'),
        (('git', 'ls-files'), 'Tracked files'),
    ]:
        code, out = run(*cmd)
        if code != 0:
            check('GIT', title, 'FAIL', out[:200])
        elif title == 'Git working tree':
            check('GIT', title, 'WARN' if out else 'PASS',
                  'Clean' if not out else f'{len(out.splitlines())} changed/untracked entries (not automatically an error)')
        elif title == 'Tracked files':
            tracked = out.splitlines()
            required = ('api/main.py', 'dashboard/app.py', 'dashboard/online_stream.py',
                        'prepare_online.py', 'validate_online_data.py', 'data.dvc')
            missing = [x for x in required if x not in tracked]
            check('GIT', title, 'PASS' if not missing else 'FAIL',
                  f'{len(tracked)} tracked; missing required: {missing}')
        else:
            check('GIT', title, 'PASS', out.replace('\n', ' | ')[:450])

    code, head = run('git', 'rev-parse', 'HEAD')
    c2, remote_head = run('git', 'rev-parse', '--verify', 'refs/remotes/github/main')
    check('GIT', 'GitHub tracking ref', 'PASS' if code == 0 and c2 == 0 and head == remote_head else 'WARN',
          'Local HEAD matches last known github/main' if code == 0 and c2 == 0 and head == remote_head
          else 'Missing or different cached github/main; may need fetch to verify')

    code, remote = run('dvc', 'remote', 'list')
    check('DVC', 'configured DVC remotes', 'PASS' if code == 0 and 'origin' in remote else 'FAIL', remote[:250])
    code, status = run('dvc', 'status', timeout=120)
    check('DVC', 'local DVC status', 'PASS' if code == 0 and 'up to date' in status.lower()
          else ('WARN' if code == 0 else 'FAIL'), status[:300])
    if check_remote:
        code, out = run('dvc', 'status', '-c', timeout=180)
        check('DVC', 'remote DVC sync', 'PASS' if code == 0 and 'in sync' in out.lower()
              else ('WARN' if code == 0 else 'FAIL'), out[:300])
    else:
        check('DVC', 'remote DVC sync', 'SKIP', 'Use --remote for read-only remote status check (may access network)')


def inspect_training():
    section('MODEL / RETRAINING / GITHUB ACTIONS')
    p = ROOT / 'mobilenet_model/mobilenet_v3_small_best.pth'
    check('ML', 'MobileNetV3 v1 checkpoint', 'PASS' if p.is_file() and p.stat().st_size > 100000 else 'FAIL',
          f'{p.relative_to(ROOT)} ({p.stat().st_size // 1024} KiB)' if p.is_file() else 'not found')
    code = ROOT / 'api/main.py'
    if code.is_file():
        content = code.read_text(encoding='utf-8-sig')
        check('ML', 'Registry/rollback logic', 'PASS' if 'get_model_version_by_alias' in content and 'load_local_model' in content else 'WARN',
              'Registry alias lookup and local fallback present' if 'get_model_version_by_alias' in content and 'load_local_model' in content else 'Review deployment logic')
        check('ML', 'ROI currently tied to test', 'WARN' if 'data" / "test' in content else 'PASS',
              'Check whether inference still reads data/test COCO; online JSON may not be used')
    training_candidates = ['retraining/retrain.py', 'retrain.py']
    found = [x for x in training_candidates if (ROOT / x).is_file()]
    check('ML', 'fine-tuning script', 'PASS' if found else 'NEXT', ', '.join(found) if found else 'Not built yet')
    evaluation_candidates = ['retraining/evaluate.py', 'evaluate.py']
    found = [x for x in evaluation_candidates if (ROOT / x).is_file()]
    check('ML', 'evaluation gate script', 'PASS' if found else 'NEXT', ', '.join(found) if found else 'Not built yet')
    workflow = ROOT / '.github/workflows/retrain.yml'
    check('ML', 'GitHub Actions retraining workflow', 'PASS' if workflow.is_file() else 'NEXT',
          'Exists' if workflow.is_file() else 'Not created yet (expected)')
    metadata = ROOT / 'kaggle/kernel-metadata.json'
    check('ML', 'Kaggle kernel metadata', 'PASS' if metadata.is_file() else 'NEXT',
          'Exists' if metadata.is_file() else 'Not created yet (expected)')
    for mod in ('torch', 'torchvision', 'mlflow', 'dagshub', 'dvc', 'PIL'):
        found = importlib.util.find_spec(mod) is not None
        check('ENV', mod, 'PASS' if found else 'WARN', 'Installed in current Python' if found else 'Not installed in current Python')
    check('ENV', 'Kaggle CLI', 'PASS' if shutil.which('kaggle') else 'NEXT',
          'Available in PATH' if shutil.which('kaggle') else 'Not found; required where kernel push runs')
    check('ENV', 'Cloud GPU/credentials', 'SKIP', 'Cannot safely verify Kaggle GPU quota or remote secrets offline')


def main():
    parser = argparse.ArgumentParser(description='Read-only MLOps retraining readiness audit')
    parser.add_argument('--remote', action='store_true', help='Also run read-only dvc status -c (network access)')
    args = parser.parse_args()
    print('SMART PARKING MLOPS - READ-ONLY PREFLIGHT')
    print('Project:', ROOT)
    print('No writes, commits, pushes, model loads or training.')
    section('PROJECT CODE')
    for f in ('api/main.py', 'dashboard/app.py', 'dashboard/online_stream.py',
              'prepare_online.py', 'validate_online_data.py'):
        parse_python(f)
    inspect_data()
    inspect_git_dvc(args.remote)
    inspect_training()
    section('SUMMARY')
    counts = Counter(x[2] for x in results)
    for state in ('PASS', 'WARN', 'FAIL', 'NEXT', 'SKIP'):
        print(f'{state}: {counts[state]}')
    if counts['FAIL']:
        print('STATUS: Fix FAILED prerequisites before Kaggle retraining.')
    else:
        print('STATUS: Core data checks passed; implement NEXT items for Kaggle retraining.')
    print('NOTE: No-correction review and test-set reuse still require human verification.')
    return 1 if counts['FAIL'] else 0


if __name__ == '__main__':
    sys.exit(main())
