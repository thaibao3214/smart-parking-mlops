"""Stage 6: read-only held-out PKLot evaluation of 3 v1/candidate pairs.

Requires torch/torchvision/Pillow and existing retraining/evaluate.py architecture.
No training, registry modification or deployment. Evaluates the same COCO crops
for each model, excludes online simulation source frames, and produces CPU latency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retraining.evaluate import load_model  # exact architecture of registered models

MODELS = ('mobilenetv3', 'resnet18', 'custom_cnn')
CANDIDATES = {
    'mobilenetv3': ('input_stage3/stage3_logs/stage3_candidate_mobilenetv3.pth', 'stage3_logs/stage3_report.json'),
    'resnet18': ('input_stage4/stage4_logs/stage4_candidate_resnet18.pth', 'stage4_logs/stage4_report.json'),
    'custom_cnn': ('input_stage4/stage4_logs/stage4_candidate_custom_cnn.pth', 'stage4_logs/stage4_report.json'),
}
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
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_path(root, name):
    p = Path(name)
    if not name or p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe image path: {name!r}')
    out = (root / p).resolve()
    if root.resolve() not in out.parents:
        raise ValueError('Image path escapes data/test')
    return out


def get_online_exclusions(path):
    """Fail closed if the project's online simulation manifest is absent."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing online manifest {path}; refusing to evaluate test frames without exclusion')
    doc = json.loads(path.read_text(encoding='utf-8'))
    if doc.get('source_split') != 'data/test':
        raise ValueError('Online manifest source_split must be data/test')
    names = doc.get('image_files')
    if not isinstance(names, list) or not names:
        raise ValueError('Online manifest image_files must be nonempty')
    excluded = {Path(name).name for name in names}
    if len(excluded) != len(names):
        raise ValueError('Duplicate online source names')
    print(f'STAGE6_ONLINE_SOURCE_EXCLUSION_PASS: {len(excluded)} frames', flush=True)
    return excluded


def read_split(test_dir, excluded, per_class, seed):
    coco_path = test_dir / '_annotations.coco.json'
    coco = json.loads(coco_path.read_text(encoding='utf-8'))
    images = {im['id']: im for im in coco['images']}
    choices = {0: [], 1: []}
    skipped = 0
    for ann in coco['annotations']:
        category = ann.get('category_id')
        if category not in (1, 2):
            continue
        im = images.get(ann.get('image_id'))
        if not im:
            continue
        name = Path(im['file_name']).name
        if name in excluded:
            skipped += 1
            continue
        imgpath = checked_path(test_dir, im['file_name'])
        if not imgpath.is_file():
            continue
        box = ann.get('bbox', [])
        if len(box) != 4:
            continue
        try:
            x, y, w, h = [float(n) for n in box]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 5 or h <= 5:
            continue
        # Bounding-box clipping is finalized using actual image dimensions at read time.
        choices[category - 1].append((str(imgpath), (x, y, w, h), category - 1, str(ann['id']), name))
    rng = random.Random(seed)
    selected = []
    for cls in (0, 1):
        choices[cls].sort(key=lambda row: (row[4], row[3]))
        rng.shuffle(choices[cls])
        if len(choices[cls]) < per_class:
            raise ValueError(f'Insufficient eligible class {cls} boxes: {len(choices[cls])} < {per_class}')
        selected.extend(choices[cls][:per_class])
    selected.sort(key=lambda row: (row[4], row[3]))
    if any(row[4] in excluded for row in selected):
        raise AssertionError('Online source leaked into evaluation')
    print(f'STAGE6_EVAL_SELECTION_PASS: {len(selected)} crops, excluded {skipped} online-source boxes', flush=True)
    return selected


class Crops(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        filename, (x, y, w, h), label, ann_id, image_name = self.rows[index]
        with Image.open(filename) as file:
            rgb = file.convert('RGB')
            x1, y1 = max(0, math.floor(x)), max(0, math.floor(y))
            x2, y2 = min(rgb.width, math.ceil(x + w)), min(rgb.height, math.ceil(y + h))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f'Invalid clipped bounding box {filename} ann={ann_id}')
            return TRANSFORM(rgb.crop((x1, y1, x2, y2))), label


def scores(truth, predicted):
    matrix = [[0, 0], [0, 0]]
    for y, p in zip(truth, predicted):
        matrix[int(y)][int(p)] += 1
    n = sum(sum(row) for row in matrix)
    if n != len(truth) or len(truth) != len(predicted) or n == 0:
        raise ValueError('Metric sample count mismatch')
    p, r, f = [], [], []
    for c in (0, 1):
        tp = matrix[c][c]
        fp = matrix[1-c][c]
        fn = matrix[c][1-c]
        prec = tp / (tp + fp) if tp + fp else 0.
        rec = tp / (tp + fn) if tp + fn else 0.
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.
        p.append(prec); r.append(rec); f.append(f1)
    return {
        'samples': n, 'accuracy': (matrix[0][0] + matrix[1][1]) / n,
        'precision_macro': statistics.mean(p), 'recall_macro': statistics.mean(r),
        'f1_macro': statistics.mean(f), 'f1_empty': f[0], 'f1_occupied': f[1],
        'confusion_matrix': matrix,
    }


def measure_latency(model, dataset, iterations):
    # Batch size 1, CPU, one configured thread, model-only forward pass excluding I/O.
    x, _ = dataset[0]
    batch = x.unsqueeze(0)
    with torch.inference_mode():
        for _ in range(10):
            model(batch)
        observed = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(batch)
            observed.append((time.perf_counter() - start) * 1000.)
    observed.sort()
    return {
        'median_ms': statistics.median(observed),
        'p95_ms': observed[min(len(observed)-1, math.ceil(.95 * len(observed))-1)],
        'measurement': 'CPU PyTorch forward only, single thread, batch=1, 64x64, fixed warmed input',
    }


def verify_reports(stage3, stage4, dataset_pointer):
    st3 = json.loads((stage3 / 'stage3_logs/stage3_report.json').read_text(encoding='utf-8'))
    st4 = json.loads((stage4 / 'stage4_logs/stage4_report.json').read_text(encoding='utf-8'))
    for report in (st3, st4):
        if report['data_dvc_sha256'] != sha256(dataset_pointer):
            raise ValueError('Current data.dvc differs from training version: refuse mixing dataset revisions')
        if report.get('online_crops_included') != 0 or report.get('test_crops_included') != 0:
            raise ValueError('Test/online crops were used to fine-tune the candidate')
        if report.get('production_approved') is not False:
            raise ValueError('Unexpected production-approved training report')
    if st3['data_dvc_sha256'] != st4['data_dvc_sha256']:
        raise ValueError('Different training data revisions')
    result = {}
    for name in MODELS:
        report = st3 if name == 'mobilenetv3' else st4['models'][name]
        candidate = (stage3 if name == 'mobilenetv3' else stage4) / Path(CANDIDATES[name][0]).relative_to('input_stage3' if name == 'mobilenetv3' else 'input_stage4')
        baseline = ROOT / BASELINES[name]
        if not candidate.is_file() or not baseline.is_file():
            raise FileNotFoundError(f'Missing checkpoint {name}: {candidate} or {baseline}')
        if sha256(candidate) != report['candidate_checkpoint_sha256']:
            raise ValueError(f'{name} candidate checkpoint SHA256 mismatch')
        if sha256(baseline) != report['v1_checkpoint_sha256']:
            raise ValueError(f'{name} baseline checkpoint SHA256 mismatch')
        result[name] = {'candidate': candidate, 'baseline': baseline, 'candidate_sha256': sha256(candidate)}
    print('STAGE6_SIX_CHECKPOINT_HASHES_PASS', flush=True)
    return result


def maybe_log_mlflow(report, out):
    token = os.getenv('DAGSHUB_TOKEN', '').strip()
    if not token:
        print('STAGE6_MLFLOW_SKIPPED: no token (offline mode)', flush=True)
        return
    if any(ord(c) < 33 or ord(c) == 127 for c in token):
        raise ValueError('Malformed DAGSHUB_TOKEN')
    import mlflow
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME', 'thaibao3214')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri(f"https://dagshub.com/{os.environ['MLFLOW_TRACKING_USERNAME']}/{os.getenv('DAGSHUB_REPO', 'smart-parking-mlops')}.mlflow")
    mlflow.set_experiment('smart-parking-stage6-independent-evaluation')
    for name, pair in report['models'].items():
        with mlflow.start_run(run_name=f'stage6-{name}-{os.environ.get("GITHUB_RUN_ID", "local")}'):
            mlflow.log_params({'architecture': name, 'data_dvc_sha256': report['data_dvc_sha256'], 'evaluation_split': 'test_excluding_online_frames', 'num_crops': report['total_samples'], 'production_approved': False})
            for kind in ('baseline', 'candidate'):
                metrics = pair[kind]
                mlflow.log_metrics({f'{kind}_{k}': v for k, v in metrics.items() if isinstance(v, (float, int))})
                mlflow.log_metrics({f'{kind}_latency_{k}': v for k, v in pair[f'{kind}_latency_cpu'].items() if isinstance(v, (float, int))})
            mlflow.log_metric('f1_nonregression_pass', int(pair['nonregression_pass']))
            mlflow.log_artifact(str(out / 'stage6_report.json'))
    print('STAGE6_MLFLOW_TRACKING_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage3', type=Path, default=ROOT / 'input_stage3')
    parser.add_argument('--stage4', type=Path, default=ROOT / 'input_stage4')
    parser.add_argument('--test-dir', type=Path, default=ROOT / 'data/test')
    parser.add_argument('--online-manifest', type=Path, default=ROOT / 'data/realtime_stream/manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'stage6_output')
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--latency-runs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if min(args.per_class, args.latency_runs, args.batch_size) < 1:
        parser.error('per-class, latency-runs and batch-size must be positive')
    torch.set_num_threads(1)
    checkpoints = verify_reports(args.stage3, args.stage4, ROOT / 'data.dvc')
    excluded = get_online_exclusions(args.online_manifest)
    data = read_split(args.test_dir.resolve(), excluded, args.per_class, args.seed)
    dataset = Crops(data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    ground_truth = [row[2] for row in data]
    result = {
        'stage': 'stage6-model-selection', 'data_dvc_sha256': sha256(ROOT / 'data.dvc'),
        'evaluation_split': 'PKLot data/test excluding online-simulation source frames',
        'source_frame_exclusions': len(excluded), 'total_samples': len(data),
        'class_mapping': {'0': 'Empty', '1': 'Occupied'},
        'cpu_threads': torch.get_num_threads(), 'models': {},
        'production_approved': False, 'selected_for_production': None,
        'caveat': 'The PKLot test split has been inspected/used earlier in this project. This is a held-out-from-finetuning comparison, not an untouched external blind test. Multiple crops from the same frame are correlated.',
    }
    preds = {}
    for name in MODELS:
        print(f'EVALUATE {name}', flush=True)
        pair = {}
        for kind in ('baseline', 'candidate'):
            model = load_model(name, checkpoints[name][kind], torch.device('cpu'))
            predictions = []
            with torch.inference_mode():
                for x, _ in loader:
                    predictions.extend(model(x).argmax(dim=1).tolist())
            metrics = scores(ground_truth, predictions)
            pair[kind] = metrics
            pair[f'{kind}_latency_cpu'] = measure_latency(model, dataset, args.latency_runs)
            preds[(name, kind)] = predictions
            del model
        # Do not accidentally choose a retrained model which got worse on this test.
        pair['nonregression_pass'] = (
            pair['candidate']['accuracy'] >= pair['baseline']['accuracy'] - .005
            and pair['candidate']['f1_macro'] >= pair['baseline']['f1_macro'] - .005
        )
        result['models'][name] = pair
        print(f"{name}: F1 v1={pair['baseline']['f1_macro']:.4f}, candidate={pair['candidate']['f1_macro']:.4f}, gate={'PASS' if pair['nonregression_pass'] else 'FAIL'}", flush=True)
    # Recommendation for review only: maximize macro-F1 then minimize single-thread CPU p95.
    qualified = [name for name in MODELS if result['models'][name]['nonregression_pass']]
    qualified.sort(key=lambda n: (-result['models'][n]['candidate']['f1_macro'], result['models'][n]['candidate_latency_cpu']['p95_ms'], n))
    result['recommended_candidate_for_review'] = qualified[0] if qualified else None
    result['review_ranking'] = qualified
    result['gate'] = 'REVIEW_ONLY_PRODUCTION_BLOCKED' if qualified else 'ALL_CANDIDATES_FAILED_PRODUCTION_BLOCKED'
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'stage6_report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (args.output / 'stage6_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_image', 'annotation_id', 'truth', 'mobilenetv3_v1', 'mobilenetv3_candidate', 'resnet18_v1', 'resnet18_candidate', 'custom_cnn_v1', 'custom_cnn_candidate'])
        for i, row in enumerate(data):
            writer.writerow([row[4], row[3], row[2]] + [preds[(name, kind)][i] for name in MODELS for kind in ('baseline', 'candidate')])
    lines = ['# Stage 6 — PKLot Test Comparison (Review Only)', '', '| Model | v1 Acc | Candidate Acc | v1 Macro F1 | Candidate Macro F1 | Candidate CPU p95 | Gate |', '|---|---:|---:|---:|---:|---:|---|']
    for name, pair in result['models'].items():
        lines.append(f"| {name} | {pair['baseline']['accuracy']:.2%} | {pair['candidate']['accuracy']:.2%} | {pair['baseline']['f1_macro']:.4f} | {pair['candidate']['f1_macro']:.4f} | {pair['candidate_latency_cpu']['p95_ms']:.2f} ms | {'PASS' if pair['nonregression_pass'] else 'FAIL'} |")
    lines += ['', f"**Candidate suggested for review:** {result['recommended_candidate_for_review'] or 'None'}", '', '**Production:** unchanged. No automatic deployment or registry promotion.', '', f"**Caveat:** {result['caveat']}", '']
    (args.output / 'stage6_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('STAGE6_METRICS_AND_CPU_LATENCY_PASS', flush=True)
    print('STAGE6_PRODUCTION_UNCHANGED_PASS', flush=True)
    if not args.offline:
        maybe_log_mlflow(result, args.output)
    print('STAGE6_END_TO_END_PASS', flush=True)


if __name__ == '__main__':
    main()
