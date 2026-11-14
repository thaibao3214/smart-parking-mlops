"""Stage 7 read-only CPU evaluation + MLflow; NEVER changes Registry production alias.
Uses same PKLot validation subset for baseline/candidate of all architectures.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retraining.evaluate import load_model

NAMES = ('mobilenetv3', 'resnet18', 'custom_cnn')
BASELINES = {
    'mobilenetv3': 'mobilenet_model/mobilenet_v3_small_best.pth',
    'resnet18': 'resnet_model/resnet18_best.pth',
    'custom_cnn': 'cnn_model/optimized_custom_cnn_best.pth',
}
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class ValDataset(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.root / row['file']) as im:
            return TRANSFORM(im.convert('RGB')), row['label']


def evaluate(model, loader):
    confusion = [[0, 0], [0, 0]]
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x).argmax(1).tolist()
            for truth, p in zip(y.tolist(), pred):
                confusion[int(truth)][p] += 1
    total = sum(map(sum, confusion))
    if not total:
        raise ValueError('Empty eval set')
    prs, recs, f1s = [], [], []
    for i in (0, 1):
        tp = confusion[i][i]
        fp = confusion[1 - i][i]
        fn = confusion[i][1 - i]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        prs.append(p); recs.append(r); f1s.append(f)
    return {'samples': total, 'accuracy': (confusion[0][0] + confusion[1][1]) / total,
            'macro_precision': statistics.mean(prs), 'macro_recall': statistics.mean(recs),
            'macro_f1': statistics.mean(f1s), 'confusion_matrix': confusion}


def latency(model, rounds=70):
    torch.set_num_threads(1)
    x = torch.zeros(1, 3, 64, 64)
    model.eval()
    times = []
    with torch.inference_mode():
        for _ in range(10):
            model(x)
        for _ in range(rounds):
            t = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {'cpu_forward_p95_ms': times[int(.95 * (len(times) - 1))],
            'cpu_forward_median_ms': statistics.median(times),
            'scope': 'PyTorch CPU forward only, 1 thread, batch 1, no preprocessing/API'}


def log_mlflow(report, out):
    import mlflow
    token = os.environ.get('DAGSHUB_TOKEN', '').strip()
    if not token or any(ord(c) < 33 or ord(c) == 127 for c in token):
        raise ValueError('Missing or malformed DAGSHUB_TOKEN')
    user = os.environ.get('DAGSHUB_USERNAME', 'thaibao3214')
    repo = os.environ.get('DAGSHUB_REPO', 'smart-parking-mlops')
    os.environ['MLFLOW_TRACKING_USERNAME'] = user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri(f'https://dagshub.com/{user}/{repo}.mlflow')
    mlflow.set_experiment('smart-parking-stage7-continuous-training')
    for name, row in report['models'].items():
        with mlflow.start_run(run_name=f"ct-{report['github_run_id']}-{name}"):
            mlflow.log_params({'model': name, 'git_commit': report['git_commit'],
                               'approved_dvc_sha256': report['approved_dvc_sha256'],
                               'data_dvc_sha256': report['data_dvc_sha256'],
                               'candidate_sha256': row['candidate_checkpoint_sha256']})
            mlflow.log_metrics({'baseline_f1': row['baseline']['macro_f1'],
                                'candidate_f1': row['candidate']['macro_f1'],
                                'baseline_accuracy': row['baseline']['accuracy'],
                                'candidate_accuracy': row['candidate']['accuracy'],
                                'candidate_cpu_p95_ms': row['latency']['cpu_forward_p95_ms'],
                                'gate_pass': int(row['gate_pass']), 'production_approved': 0})
            mlflow.set_tags({'stage': '7', 'decision': 'REVIEW_ONLY', 'production_approved': 'false'})
            mlflow.log_artifact(str(out / 'stage7_report.json'), artifact_path='evaluation')
            print('STAGE7_MLFLOW_MODEL_LOGGED:', name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='.stage7_dataset')
    parser.add_argument('--kaggle-output', default='stage7_logs')
    parser.add_argument('--out', default='stage7_output')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = Path(args.dataset)
    output = Path(args.kaggle_output)
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    train_report = json.loads((output / 'stage7_train_report.json').read_text(encoding='utf-8'))
    if str(train_report['github_run_id']) != str(os.environ['GITHUB_RUN_ID']) or train_report['git_commit'] != os.environ['GITHUB_SHA']:
        raise ValueError('Run ID or commit mismatch')
    for key in ('approved_dvc_sha256', 'data_dvc_sha256'):
        if train_report[key] != manifest[key]:
            raise ValueError(f'{key} mismatch')
    if set(train_report['models']) != set(NAMES) or train_report['production_approved'] is not False:
        raise ValueError('Untrusted training report')
    extract = out / 'validation'
    extract.mkdir(exist_ok=True)
    with zipfile.ZipFile(base / 'payload.zip') as archive:
        for row in manifest['valid']:
            name = row['file']
            if not name.startswith('valid/') or len(Path(name).parts) != 2:
                raise ValueError('Unsafe validation entry')
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('Validation image hash mismatch')
            dest = extract / name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(raw)
    loader = DataLoader(ValDataset(extract, manifest['valid']), batch_size=32, shuffle=False)
    rows = {}
    for name in NAMES:
        old = Path(BASELINES[name])
        new = output / f'stage7_candidate_{name}.pth'
        entry = train_report['models'][name]
        if sha256(old) != manifest['checkpoints'][name]['sha256'] or sha256(new) != entry['candidate_checkpoint_sha256']:
            raise ValueError(f'{name} checkpoint SHA256 mismatch')
        before = load_model(name, old, 'cpu')
        after = load_model(name, new, 'cpu')
        baseline = evaluate(before, loader)
        candidate = evaluate(after, loader)
        l = latency(after)
        # Nonregression + a minimum validation quality gate; all are review-only.
        passed = (candidate['accuracy'] >= baseline['accuracy'] and
                  candidate['macro_f1'] >= baseline['macro_f1'] and
                  candidate['macro_f1'] >= .98 and l['cpu_forward_p95_ms'] <= 40)
        rows[name] = {'baseline': baseline, 'candidate': candidate, 'latency': l,
                      'gate_pass': passed, 'candidate_checkpoint_sha256': sha256(new)}
        print(f'STAGE7_{name.upper()}_GATE: {passed} baseline_f1={baseline["macro_f1"]:.5f} candidate_f1={candidate["macro_f1"]:.5f}')
    allowed = [name for name in NAMES if rows[name]['gate_pass']]
    suggested = min(allowed, key=lambda n: (-rows[n]['candidate']['macro_f1'],
                                             rows[n]['latency']['cpu_forward_p95_ms'])) if allowed else None
    report = {'stage': 'stage7-continuous-training-gate', 'github_run_id': train_report['github_run_id'],
              'git_commit': train_report['git_commit'], 'approved_dvc_sha256': train_report['approved_dvc_sha256'],
              'data_dvc_sha256': train_report['data_dvc_sha256'], 'models': rows,
              'suggested_for_review': suggested, 'gate': 'REVIEW_ONLY_PRODUCTION_BLOCKED',
              'production_approved': False, 'registry_alias_changed': False,
              'evaluation_split': 'PKLot data/valid (not an untouched external blind test)'}
    (out / 'stage7_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Stage 7 — Continuous Training / Evaluation Gate', '',
             '| Model | v1 macro F1 | Candidate macro F1 | CPU p95 | Gate |',
             '|---|---:|---:|---:|---|']
    for n, r in rows.items():
        lines.append(f"| {n} | {r['baseline']['macro_f1']:.4f} | {r['candidate']['macro_f1']:.4f} | {r['latency']['cpu_forward_p95_ms']:.2f} ms | {'PASS' if r['gate_pass'] else 'FAIL'} |")
    lines += ['', f'**Candidate suggested for review:** {suggested or "none"}', '',
              '**Production:** UNCHANGED — MLflow alias `production` not modified.', '',
              'Validation uses PKLot validation crops. Final deployment requires human approval and an additional deployment workflow.']
    (out / 'stage7_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('STAGE7_EVALUATION_GATE_PASS')
    print('STAGE7_PRODUCTION_UNCHANGED_PASS')
    if not args.offline:
        log_mlflow(report, out)
        print('STAGE7_MLFLOW_TRACKING_PASS')
    print('STAGE7_END_TO_END_PASS')


if __name__ == '__main__':
    main()
