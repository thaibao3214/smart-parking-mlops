"""Post-process TEST_ONLY gate metrics and log under a separate MLflow experiment."""
import json
import os
from pathlib import Path


def main():
    root = Path('stage7_test_output')
    source = root / 'stage7_report.json'
    report = json.loads(source.read_text(encoding='utf-8'))
    train = json.loads(Path('stage7_test_logs/stage7_train_report.json').read_text(encoding='utf-8'))
    if train.get('stage') != 'stage7-test-only' or train.get('test_only') is not True:
        raise ValueError('Not a test-only training result')
    if str(report['github_run_id']) != os.environ['GITHUB_RUN_ID'] or report['git_commit'] != os.environ['GITHUB_SHA']:
        raise ValueError('Stale gate report')
    report.update({
        'stage': 'stage7-test-only-gate', 'test_only': True,
        'gate': 'TEST_ONLY_NO_PROMOTION', 'suggested_for_review': None,
        'production_approved': False, 'registry_alias_changed': False,
        'warning': 'PKLot train -> PKLot valid technical test; do not treat as live data or production evaluation',
    })
    path = root / 'stage7_test_report.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (root / 'stage7_summary.md').open('a', encoding='utf-8') as fp:
        fp.write('\n\n**TEST ONLY: PKLot train/valid, NOT new online data. No Production promotion permitted.**\n')
    import mlflow
    token = os.environ.get('DAGSHUB_TOKEN', '').strip()
    if not token or any(ord(ch) < 33 or ord(ch) == 127 for ch in token):
        raise ValueError('Missing/malformed DAGSHUB_TOKEN')
    username = os.environ.get('DAGSHUB_USERNAME', 'thaibao3214')
    repo = os.environ.get('DAGSHUB_REPO', 'smart-parking-mlops')
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri(f'https://dagshub.com/{username}/{repo}.mlflow')
    mlflow.set_experiment('smart-parking-stage7-TEST-ONLY')
    for name, row in report['models'].items():
        with mlflow.start_run(run_name=f'test-only-{report["github_run_id"]}-{name}'):
            mlflow.set_tags({'stage': '7', 'test_only': 'true', 'origin': 'PKLot train',
                             'decision': 'NO_PROMOTION', 'production_approved': 'false'})
            mlflow.log_params({'model': name, 'git_commit': report['git_commit'],
                               'data_dvc_sha256': report['data_dvc_sha256']})
            mlflow.log_metrics({'baseline_macro_f1': row['baseline']['macro_f1'],
                                'candidate_macro_f1': row['candidate']['macro_f1'],
                                'baseline_accuracy': row['baseline']['accuracy'],
                                'candidate_accuracy': row['candidate']['accuracy'],
                                'cpu_p95_ms': row['latency']['cpu_forward_p95_ms'],
                                'nonregression_gate_pass': int(row['gate_pass'])})
            mlflow.log_artifact(str(path), artifact_path='test-only')
            print(f'STAGE7_TEST_ONLY_MLFLOW_LOGGED: {name}')
    print('STAGE7_TEST_ONLY_MLFLOW_PASS')
    print('STAGE7_TEST_ONLY_PRODUCTION_UNCHANGED_PASS')


if __name__ == '__main__':
    main()
