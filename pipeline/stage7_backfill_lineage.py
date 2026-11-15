"""Backfill MLflow Datasets on EXISTING Stage7 QA v3 run (no new registry version)."""
import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

from stage7_auto_register_qa import log_dataset_lineage

MODEL = 'SmartParking-MobileNetV3-Small'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-zip', type=Path, required=True,
                        help='Download artifact zip from GitHub run 43652934712')
    parser.add_argument('--version', default='3', help='MLflow QA version (default: 3)')
    parser.add_argument('--run-id', default='4d049ffccee0470daf8c3330584a8562',
                        help='Existing QA MLflow run ID')
    parser.add_argument('--execute', action='store_true', help='Actually log datasets')
    args = parser.parse_args()

    if not zipfile.is_zipfile(args.artifact_zip):
        raise ValueError('Invalid GitHub Actions artifact ZIP')
    with zipfile.ZipFile(args.artifact_zip) as z:
        paths = [n for n in z.namelist() if n.endswith('.stage7_test_dataset/manifest.json')]
        if len(paths) != 1:
            raise ValueError('Expected exactly one stage7 dataset manifest')
        manifest = json.loads(z.read(paths[0]))

    if (str(manifest.get('github_run_id')) != '43652934712'
            or manifest.get('stage') != 'stage7-test-only'
            or manifest.get('origin') != 'pklot_train_test_only'
            or manifest.get('test_only') is not True
            or manifest.get('production_approved') is not False
            or len(manifest.get('train', [])) != 2000
            or len(manifest.get('valid', [])) != 256):
        raise ValueError('Artifact is not the verified Stage7 train simulation run')

    # Verify DVC pointers from the EXACT Git commit used by GitHub Actions.
    # Do not compare local working-tree files: DVC/Git and Windows line endings
    # can change them after the original training run.
    commit = str(manifest['git_commit'])
    if (len(commit) != 40 and len(commit) != 7) or any(
            c not in '0123456789abcdef' for c in commit.lower()):
        raise ValueError('Invalid training Git commit in manifest')
    for file, key in [('confirmed_train_sim.dvc', 'online_dvc_sha256'),
                      ('simulation_train.dvc', 'sim_dvc_sha256'),
                      ('data.dvc', 'data_dvc_sha256')]:
        try:
            original = subprocess.run(
                ['git', 'show', f'{commit}:{file}'],
                check=True, capture_output=True,
            ).stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                f'Cannot read historical DVC pointer {file} at commit {commit}. '
                'Run from the project root with the original commit available.'
            ) from exc
        actual = hashlib.sha256(original).hexdigest()
        if actual != manifest[key]:
            raise ValueError(
                f'Historical DVC pointer mismatch: {file}; '
                f'expected {manifest[key]}, got {actual}'
            )
    print('STAGE7_HISTORICAL_DVC_POINTERS_PASS:', commit)

    print('STAGE7_LINEAGE_ARTIFACT_VERIFIED_PASS: 43652934712')
    print('Train: 2000, Validation: 256')
    if not args.execute:
        print('STAGE7_LINEAGE_DRY_RUN_NO_MLFLOW_CHANGES')
        return

    import mlflow
    from mlflow.tracking import MlflowClient
    token = os.environ.get('DAGSHUB_TOKEN', '').strip()
    if not token or any(ord(c) < 33 or ord(c) == 127 for c in token):
        raise ValueError('Missing/malformed DAGSHUB_TOKEN')
    user = os.environ.get('DAGSHUB_USERNAME', 'thaibao3214')
    repo = os.environ.get('DAGSHUB_REPO', 'smart-parking-mlops')
    uri = f'https://dagshub.com/{user}/{repo}.mlflow'
    os.environ['MLFLOW_TRACKING_USERNAME'] = user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()

    version = client.get_model_version(MODEL, args.version)
    run = client.get_run(args.run_id)
    if (version.run_id != args.run_id
            or (version.tags or {}).get('github_run_id') != '43652934712'
            or run.data.tags.get('github_run_id') != '43652934712'
            or run.data.tags.get('review_status') != 'QA_ONLY'
            or run.data.tags.get('production_approved') != 'false'
            or run.data.tags.get('git_commit') != manifest['git_commit']):
        raise ValueError('QA model version and run are not the expected candidate')
    protected = client.get_model_version_by_alias(MODEL, 'production').version
    if str(protected) != '1':
        raise ValueError('Production alias is not v1; refusing backfill')
    if run.inputs and run.inputs.dataset_inputs:
        print('STAGE7_LINEAGE_ALREADY_PRESENT: run', args.run_id)
        return

    with mlflow.start_run(run_id=args.run_id):
        log_dataset_lineage(mlflow, manifest)
        mlflow.set_tag('stage7_batch_data_lineage', 'DVC confirmed_train_sim and PKLot valid')

    if str(client.get_model_version_by_alias(MODEL, 'production').version) != str(protected):
        raise RuntimeError('Production alias unexpectedly changed')
    updated = client.get_run(args.run_id)
    if not updated.inputs or len(updated.inputs.dataset_inputs) < 2:
        raise RuntimeError('MLflow input datasets were not persisted')
    print('STAGE7_MLFLOW_DATASETS_BACKFILLED_PASS:', args.run_id)
    print('STAGE7_PRODUCTION_ALIAS_UNCHANGED:', protected)


if __name__ == '__main__':
    main()
