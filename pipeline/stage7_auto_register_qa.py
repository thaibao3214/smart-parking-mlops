"""Register Stage 7 TEST_ONLY winning model as a QA registry version, no aliases.

GitHub Actions invokes this after stage7_test_log.py. Checkpoints must match
both Kaggle training report and independent CPU gate report SHA256 digests.
"""
import hashlib
import json
import os
from pathlib import Path

NAMES = {'mobilenetv3': 'SmartParking-MobileNetV3-Small',
         'resnet18': 'SmartParking-ResNet18',
         'custom_cnn': 'SmartParking-CustomCNN'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(root=Path('.')):
    report = json.loads((root / 'stage7_test_output/stage7_test_report.json').read_text(encoding='utf-8'))
    train = json.loads((root / 'stage7_test_logs/stage7_train_report.json').read_text(encoding='utf-8'))
    manifest = json.loads((root / '.stage7_test_dataset/manifest.json').read_text(encoding='utf-8'))
    expected_run = os.environ.get('GITHUB_RUN_ID')
    expected_sha = os.environ.get('GITHUB_SHA')
    if not expected_run or not expected_sha:
        raise ValueError('Run only with GITHUB_RUN_ID and GITHUB_SHA')
    for record in (report, train, manifest):
        if str(record.get('github_run_id')) != expected_run or record.get('git_commit') != expected_sha:
            raise ValueError('Stale GitHub run/commit mismatch')
        if record.get('test_only') is not True or record.get('production_approved') is not False:
            raise ValueError('Untrusted or not strictly test-only result')
    if (report.get('gate') != 'TEST_ONLY_NO_PROMOTION' or
            report.get('registry_alias_changed') is not False or
            train.get('stage') != 'stage7-test-only' or
            manifest.get('origin') != 'pklot_train_test_only' or
            manifest.get('test_only') is not True or
            manifest.get('production_approved') is not False):
        raise ValueError('Gate/provenance contract violated')
    if len(manifest.get('train', [])) != 2000 or len(manifest.get('valid', [])) != 256:
        raise ValueError('Unexpected training/holdout sizes')
    if (train.get('train_samples') != 2000 or train.get('valid_samples') != 256 or
            report.get('data_dvc_sha256') != manifest.get('data_dvc_sha256') or
            train.get('online_dvc_sha256') != manifest.get('online_dvc_sha256')):
        raise ValueError('Dataset provenance mismatch')
    for name in NAMES:
        cp = root / f'stage7_test_logs/stage7_candidate_{name}.pth'
        checksum = digest(cp)
        if (checksum != report['models'][name]['candidate_checkpoint_sha256'] or
                checksum != train['models'][name]['candidate_checkpoint_sha256']):
            raise ValueError(f'Checkpoint integrity mismatch: {name}')
    candidates = [n for n in NAMES if report['models'][n].get('gate_pass') is True]
    winners = sorted(candidates, key=lambda n: (
        -float(report['models'][n]['candidate']['macro_f1']),
        float(report['models'][n]['latency']['cpu_forward_p95_ms']), n))
    winner = report.get('suggested_for_review')
    if winner != (winners[0] if winners else None):
        raise ValueError(f'Untrusted winner: {winner}; expected {winners[:1]}')
    if winner is not None and not report['models'][winner]['gate_pass']:
        raise ValueError('Winner did not pass CPU evaluation gate')
    return report, train, manifest, winner


def register(report, train, manifest, winner, root=Path('.')):
    import torch
    import mlflow
    from mlflow.tracking import MlflowClient
    import importlib.util
    source = Path(__file__).resolve().parent.parent / 'kaggle' / 'stage7_test_train.py'
    spec = importlib.util.spec_from_file_location('stage7_model_definitions', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_model = module.build_model

    token = os.environ.get('DAGSHUB_TOKEN', '').strip()
    if not token or any(ord(c) < 33 or ord(c) == 127 for c in token):
        raise ValueError('Missing DAGSHUB_TOKEN')
    user = os.environ.get('DAGSHUB_USERNAME', 'thaibao3214')
    repo = os.environ.get('DAGSHUB_REPO', 'smart-parking-mlops')
    uri = f'https://dagshub.com/{user}/{repo}.mlflow'
    os.environ['MLFLOW_TRACKING_USERNAME'] = user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()
    model_name = NAMES[winner]
    checkpoint = root / f'stage7_test_logs/stage7_candidate_{winner}.pth'
    sha = digest(checkpoint)

    # Every run must preserve any existing production alias, regardless of winner.
    protected = NAMES['mobilenetv3']
    protected_version = client.get_model_version_by_alias(protected, 'production').version
    for version in client.search_model_versions(f"name='{model_name}'"):
        if (version.tags or {}).get('stage7_candidate_sha256') == sha:
            print('STAGE7_QA_ALREADY_REGISTERED:', model_name, version.version)
            return

    model = build_model(winner)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        state = state.get('state_dict', state.get('model_state_dict', state))
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        assert tuple(model(torch.zeros(1, 3, 64, 64)).shape) == (1, 2)

    row = report['models'][winner]
    mlflow.set_experiment('smart-parking-stage7-TEST-ONLY-QA')
    with mlflow.start_run(run_name=f"auto-qa-{report['github_run_id']}-{winner}") as run:
        mlflow.set_tags({'stage': '7', 'test_only': 'true',
                         'review_status': 'QA_ONLY', 'data_origin': 'pklot_train_simulation',
                         'production_approved': 'false',
                         'github_run_id': str(report['github_run_id']),
                         'git_commit': report['git_commit'],
                         'stage7_candidate_sha256': sha})
        mlflow.log_params({'model': winner, 'train_samples': len(manifest['train']),
                           'valid_samples': len(manifest['valid']),
                           'online_dvc_sha256': manifest['online_dvc_sha256']})
        mlflow.log_metrics({'candidate_macro_f1': row['candidate']['macro_f1'],
                            'candidate_accuracy': row['candidate']['accuracy'],
                            'cpu_forward_p95_ms': row['latency']['cpu_forward_p95_ms']})
        mlflow.pytorch.log_model(model, artifact_path='model')
        registered = mlflow.register_model(f'runs:/{run.info.run_id}/model', model_name)
    for key, value in {'review_status': 'QA_ONLY', 'test_only': 'true',
                       'production_approved': 'false',
                       'data_origin': 'pklot_train_simulation',
                       'github_run_id': str(report['github_run_id']),
                       'git_commit': report['git_commit'],
                       'stage7_candidate_sha256': sha}.items():
        client.set_model_version_tag(model_name, registered.version, key, value)
    after = client.get_model_version_by_alias(protected, 'production').version
    if str(after) != str(protected_version):
        raise RuntimeError('CRITICAL: Production alias was unexpectedly changed')
    print('STAGE7_QA_VERSION_REGISTERED:', model_name, registered.version)
    print('STAGE7_PRODUCTION_ALIAS_UNCHANGED:', after)


def main():
    report, train, manifest, winner = verify()
    print('STAGE7_AUTO_QA_VERIFICATION_PASS')
    if winner is None:
        print('STAGE7_NO_ELIGIBLE_WINNER_NO_QA_REGISTRATION')
        return
    print('STAGE7_SELECTED_WINNER:', winner)
    register(report, train, manifest, winner)


if __name__ == '__main__':
    main()
