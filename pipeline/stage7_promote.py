"""EXPLICITLY MANUAL MobileNetV3 MLflow Registry promotion / rollback.

Never called by Stage 7 GitHub Actions. Requires human-entered checksum and run ID.
Uses model name/alias currently read by main.py FastAPI ModelManager.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retraining.evaluate import load_model

MODEL_NAME = 'SmartParking-MobileNetV3-Small'
ALIAS = 'production'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mlflow_client():
    import mlflow
    from mlflow.tracking import MlflowClient
    token = os.environ.get('DAGSHUB_TOKEN', '').strip()
    if not token or any(ord(c) < 33 or ord(c) == 127 for c in token):
        raise ValueError('Set clean DAGSHUB_TOKEN environment variable')
    user = os.environ.get('DAGSHUB_USERNAME', 'thaibao3214')
    repo = os.environ.get('DAGSHUB_REPO', 'smart-parking-mlops')
    os.environ['MLFLOW_TRACKING_USERNAME'] = user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri(f'https://dagshub.com/{user}/{repo}.mlflow')
    return mlflow, MlflowClient()


def verify_model(model):
    model.eval()
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, 64, 64))
    if tuple(out.shape) != (1, 2) or not torch.isfinite(out).all():
        raise RuntimeError('Loaded model has invalid output')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', help='Stage 7 evaluation stage7_report.json')
    p.add_argument('--candidate', help='Stage 7 candidate MobileNetV3 state_dict')
    p.add_argument('--approve-sha256', help='Human-reviewed expected checkpoint SHA256')
    p.add_argument('--approve-run-id', help='Human-reviewed GitHub Actions run ID')
    p.add_argument('--rollback-version', help='Previously known safe Registry model version')
    p.add_argument('--confirm-production', action='store_true', help='Explicit permission to change production alias')
    args = p.parse_args()
    if not args.confirm_production:
        raise SystemExit('REFUSED: --confirm-production required. Production remains unchanged.')
    mlflow, client = mlflow_client()
    try:
        old = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
    except Exception as e:
        raise SystemExit(f'REFUSED: cannot resolve current production alias: {e}')
    if args.rollback_version:
        if args.candidate or args.report:
            raise SystemExit('Rollback mode cannot accept a new candidate')
        target = client.get_model_version(MODEL_NAME, args.rollback_version)
        if str(target.version) == str(old):
            print('Production already at requested version', old)
            return
        loaded = mlflow.pytorch.load_model(f'models:/{MODEL_NAME}/{target.version}', map_location='cpu')
        verify_model(loaded)
        client.set_registered_model_alias(MODEL_NAME, ALIAS, str(target.version))
        print('STAGE7_MANUAL_ROLLBACK_PASS:', 'old=', old, 'new=', target.version)
        return
    if not all((args.report, args.candidate, args.approve_sha256, args.approve_run_id)):
        raise SystemExit('Promotion requires --report --candidate --approve-sha256 --approve-run-id')
    report = json.loads(Path(args.report).read_text(encoding='utf-8'))
    if report.get('stage') != 'stage7-continuous-training-gate' or report.get('production_approved') is not False:
        raise ValueError('Not a review-only Stage 7 report')
    if str(report.get('github_run_id')) != str(args.approve_run_id):
        raise ValueError('Approval run ID mismatch')
    if report.get('suggested_for_review') != 'mobilenetv3':
        raise ValueError('Only recommended MobileNetV3 can be promoted by this tool')
    row = report['models']['mobilenetv3']
    if not row['gate_pass']:
        raise ValueError('Candidate failed evaluation gate')
    actual = digest(args.candidate)
    if actual != args.approve_sha256 or actual != row['candidate_checkpoint_sha256']:
        raise ValueError('SHA256 mismatch: no production change')
    model = load_model('mobilenetv3', Path(args.candidate), 'cpu')
    verify_model(model)
    # Log a new full MLflow PyTorch model (not merely a .pth artifact).
    with mlflow.start_run(run_name=f'manual-approved-stage7-{args.approve_run_id}') as run:
        mlflow.set_tags({'stage': '7', 'approval': 'manual', 'source_github_run_id': str(args.approve_run_id),
                         'source_sha256': actual, 'previous_production_version': str(old)})
        mlflow.log_artifact(args.report, artifact_path='approval')
        mlflow.pytorch.log_model(model, artifact_path='model')
        run_id = run.info.run_id
    version = mlflow.register_model(f'runs:/{run_id}/model', MODEL_NAME, await_registration_for=300)
    # Read back from Registry and execute a batch-1 sanity check BEFORE alias switch.
    loaded = mlflow.pytorch.load_model(f'models:/{MODEL_NAME}/{version.version}', map_location='cpu')
    verify_model(loaded)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, str(version.version))
    print('STAGE7_MANUAL_PRODUCTION_PROMOTION_PASS')
    print('MODEL:', MODEL_NAME, 'OLD:', old, 'NEW:', version.version)
    print('ROLLBACK COMMAND: python pipeline/stage7_promote.py --rollback-version', old, '--confirm-production')
    print('Restart/reload local FastAPI service and verify /health and inference')


if __name__ == '__main__':
    main()
