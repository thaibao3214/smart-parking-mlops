"""Stage 5: audit Stage 3/4 GitHub artifacts, log to DagsHub MLflow, never deploy.

Usage:
    python pipeline/stage5_evaluate.py --stage3 /path/to/unpacked-stage3 --stage4 /path/to/unpacked-stage4
    python pipeline/stage5_evaluate.py --stage3 stage3.zip --stage4 stage4.zip --offline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile

RUNS = {'mobilenetv3': '43458044027', 'resnet18': '43479352874', 'custom_cnn': '43479352874'}
DATA_MIN = 64
BASELINE_ALLOWED_DROP = 0.0


def digest_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class Source:
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.z = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.z and self.z.testzip() is not None:
            raise ValueError(f'Corrupted ZIP: {self.path}')

    def read(self, name: str) -> bytes:
        if self.z:
            return self.z.read(name)
        return (self.path / name).read_bytes()

    def sha256(self, name: str) -> str:
        if self.z:
            h = hashlib.sha256()
            with self.z.open(name) as f:
                for b in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(b)
            return h.hexdigest()
        return digest_file(self.path / name)

    def close(self):
        if self.z:
            self.z.close()


def required_fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'Invalid {name}: {value!r}')
    return float(value)


def validate_source(s, stage):
    prefix = 'stage3' if stage == 3 else 'stage4'
    report = json.loads(s.read(f'{prefix}_logs/{prefix}_report.json'))
    manifest = json.loads(s.read('.stage2_dataset/manifest.json'))
    status = s.read(f'{prefix}_logs/status.txt').decode('utf-8').strip()
    assert status == 'COMPLETE', f'Stage {stage} did not complete: {status}'
    assert report['stage'] == ('stage3-finetune' if stage == 3 else 'stage4-two-models')
    assert report['github_run_id'] == RUNS['mobilenetv3' if stage == 3 else 'resnet18']
    assert report['github_run_id'] == str(manifest['github_run_id'])
    assert report['git_commit'] == manifest['git_commit']
    assert report['data_dvc_sha256'] == manifest['data_dvc_sha256']
    assert report['online_crops_included'] == manifest['online_crops_included'] == 0
    assert report['test_crops_included'] == manifest['test_crops_included'] == 0
    assert report['production_approved'] is False
    assert len(manifest['train']) == report['train_samples'] == 128
    assert len(manifest['valid']) == report['valid_samples'] == DATA_MIN
    assert len({x['file'] for x in manifest['valid']}) == DATA_MIN
    assert isinstance(report.get('gpu'), str) and report['gpu']
    return report, manifest


def make_row(s, name, report, stage):
    p = report if stage == 3 else report['models'][name]
    filename = f'stage{stage}_logs/stage{stage}_candidate_{name}.pth'
    expected_hash = p['candidate_checkpoint_sha256']
    actual_hash = s.sha256(filename)
    assert actual_hash == expected_hash, f'{name}: checkpoint SHA256 mismatch'
    baseline = required_fraction(p['baseline_valid_accuracy'], f'{name} baseline')
    candidate = required_fraction(p['candidate_valid_accuracy'], f'{name} candidate')
    train_loss = float(p['train_loss'])
    assert math.isfinite(train_loss) and train_loss >= 0
    passed = candidate >= baseline - BASELINE_ALLOWED_DROP and candidate >= 0.98
    return {
        'model': name, 'source_stage': stage, 'source_run_id': report['github_run_id'],
        'source_commit': report['git_commit'], 'data_dvc_sha256': report['data_dvc_sha256'],
        'gpu': report['gpu'], 'epochs': int(report['epochs']),
        'train_samples': report['train_samples'], 'valid_samples': report['valid_samples'],
        'baseline_valid_accuracy': baseline, 'candidate_valid_accuracy': candidate,
        'train_loss': train_loss, 'candidate_checkpoint_sha256': actual_hash,
        'v1_checkpoint_sha256': p['v1_checkpoint_sha256'],
        'accuracy_nonregression_pass': passed,
        'f1_measured': False, 'cpu_latency_measured': False,
        'production_eligible': False,
    }


def write_summary(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Accuracy ties on 64 samples; no reliable winner without independent evaluation.
    best = max(x['candidate_valid_accuracy'] for x in rows)
    leaders = [x['model'] for x in rows if x['candidate_valid_accuracy'] == best]
    report = {
        'stage': 'stage5-evaluation-gate', 'model_count': len(rows),
        'same_data_dvc_version': len({x['data_dvc_sha256'] for x in rows}) == 1,
        'all_accuracy_nonregression_pass': all(x['accuracy_nonregression_pass'] for x in rows),
        'leaderboard_by_accuracy': leaders,
        'selected_for_production': None, 'production_approved': False,
        'gate': 'PROVISIONAL_PASS_PRODUCTION_BLOCKED' if all(x['accuracy_nonregression_pass'] for x in rows) else 'ACCURACY_GATE_FAILED_PRODUCTION_BLOCKED',
        'blocking_reasons': [
            'Validation has only 64 samples; generalization not established',
            'No per-sample predictions: precision, recall and F1 not available',
            'CPU inference latency has not been measured',
            'Independent leakage-safe evaluation is required before production approval',
        ],
        'models': rows,
    }
    out = output_dir / 'stage5_report.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# Stage 5 — MLflow / Evaluation Gate', '', '| Model | Baseline | Candidate | Nonregression |', '|---|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['baseline_valid_accuracy']:.4%} | {row['candidate_valid_accuracy']:.4%} | {'PASS' if row['accuracy_nonregression_pass'] else 'FAIL'} |")
    lines += ['', f"**Gate:** {report['gate']}", '', '**Production:** KHÔNG cập nhật. Chưa đo F1, latency CPU và chưa kiểm thử tập độc lập.', '', f"**Các mô hình đồng hạng Accuracy:** {', '.join(leaders)}", '']
    (output_dir / 'stage5_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def log_mlflow(report, output_dir):
    import mlflow
    from mlflow.tracking import MlflowClient

    token = os.environ.get('DAGSHUB_TOKEN', '').strip()
    username = os.environ.get('DAGSHUB_USERNAME', 'thaibao3214').strip()
    repo = os.environ.get('DAGSHUB_REPO', 'smart-parking-mlops').strip()
    if not token or not username or not repo:
        raise RuntimeError('MLflow needs DAGSHUB_TOKEN, DAGSHUB_USERNAME and DAGSHUB_REPO')
    if any(ord(c) < 33 or ord(c) == 127 for c in token):
        raise RuntimeError('Malformed DAGSHUB_TOKEN')
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    uri = f'https://dagshub.com/{username}/{repo}.mlflow'
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment('smart-parking-stage5-evaluation')
    client = MlflowClient()
    for row in report['models']:
        with mlflow.start_run(run_name=f"stage5-{row['model']}-source-{row['source_run_id']}") as run:
            mlflow.log_params({
                'architecture': row['model'], 'source_stage': row['source_stage'],
                'source_run_id': row['source_run_id'], 'source_commit': row['source_commit'],
                'data_dvc_sha256': row['data_dvc_sha256'],
                'candidate_checkpoint_sha256': row['candidate_checkpoint_sha256'],
                'v1_checkpoint_sha256': row['v1_checkpoint_sha256'],
                'gpu': row['gpu'], 'epochs': row['epochs'],
                'train_samples': row['train_samples'], 'valid_samples': row['valid_samples'],
            })
            mlflow.log_metrics({
                'baseline_valid_accuracy': row['baseline_valid_accuracy'],
                'candidate_valid_accuracy': row['candidate_valid_accuracy'],
                'train_loss': row['train_loss'],
                'accuracy_nonregression_pass': int(row['accuracy_nonregression_pass']),
                'production_eligible': 0,
            })
            mlflow.set_tags({'stage': '5', 'gate': report['gate'], 'production_approved': 'false', 'evidence': 'stage3_stage4_artifacts'})
            mlflow.log_artifact(str(output_dir / 'stage5_report.json'), artifact_path='evaluation')
            print('MLFLOW_MODEL_LOGGED:', row['model'], run.info.run_id)
    with mlflow.start_run(run_name='stage5-gate-summary') as run:
        mlflow.log_metrics({'model_count': 3, 'all_accuracy_nonregression_pass': int(report['all_accuracy_nonregression_pass']), 'production_approved': 0})
        mlflow.set_tags({'stage': '5', 'gate': report['gate'], 'production_approved': 'false'})
        mlflow.log_artifact(str(output_dir / 'stage5_report.json'), artifact_path='evaluation')
        mlflow.log_artifact(str(output_dir / 'stage5_summary.md'), artifact_path='evaluation')
        print('MLFLOW_GATE_LOGGED:', run.info.run_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage3', required=True)
    ap.add_argument('--stage4', required=True)
    ap.add_argument('--output', default='stage5_output')
    ap.add_argument('--offline', action='store_true', help='Validate and produce report without MLflow uploads')
    args = ap.parse_args()
    sources = [Source(args.stage3), Source(args.stage4)]
    try:
        r3, m3 = validate_source(sources[0], 3)
        r4, m4 = validate_source(sources[1], 4)
        assert m3['data_dvc_sha256'] == m4['data_dvc_sha256'], 'Different DVC data version'
        assert m3['train'] == m4['train'], 'Train subsets differ'
        assert m3['valid'] == m4['valid'], 'Validation subsets differ'
        print('STAGE5_ARTIFACT_INTEGRITY_PASS')
        print('STAGE5_IDENTICAL_VALIDATION_SPLIT_PASS')
        rows = [make_row(sources[0], 'mobilenetv3', r3, 3)]
        rows.extend(make_row(sources[1], n, r4, 4) for n in ('resnet18', 'custom_cnn'))
        print('STAGE5_THREE_CANDIDATES_CHECKSUM_PASS')
        out = Path(args.output)
        report = write_summary(rows, out)
        for row in rows:
            print(f"{row['model']}: {row['baseline_valid_accuracy']:.4%} -> {row['candidate_valid_accuracy']:.4%}; accuracy_gate={row['accuracy_nonregression_pass']}")
        print('STAGE5_GATE:', report['gate'])
        print('STAGE5_PRODUCTION_UNCHANGED_PASS')
        if not args.offline:
            log_mlflow(report, out)
            print('STAGE5_MLFLOW_TRACKING_PASS')
        print('STAGE5_END_TO_END_PASS')
    finally:
        for s in sources:
            s.close()


if __name__ == '__main__':
    main()
