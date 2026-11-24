"""Local Stage 7 launcher. Use only from the protected Streamlit control panel.

Runs existing immutable-batch TEST_ONLY pipeline, does not change Production.
Keep job artifacts outside the repository so no accidental Git commit includes logs.
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def job_dir():
    return Path.home() / '.smartparking_stage7_control'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(file, **values):
    tmp = file.with_suffix('.tmp')
    tmp.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(file)



GITHUB_REPO = 'thaibao3214/smart-parking-mlops'
WORKFLOW = 'stage7_test_only.yml'


def github_json(method, path, token, payload=None):
    url = 'https://api.github.com/repos/' + GITHUB_REPO + path
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={'Authorization': 'Bearer ' + token,
                 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28',
                 'User-Agent': 'smart-parking-streamlit-stage7',
                 **({'Content-Type': 'application/json'} if data is not None else {})})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        # Don't print response body or credentials to the dashboard log.
        raise RuntimeError(f'GitHub API returned HTTP {exc.code} for {method} {path.split("?")[0]}') from None
    except urllib.error.URLError:
        raise RuntimeError('Không kết nối được GitHub API.') from None


def dispatch_test_only(root, batch, batch_id, out):
    """Re-dispatch one already-committed immutable TEST_ONLY batch (no push/promotion)."""
    if not re.fullmatch(r'[0-9a-f]{20}', batch_id):
        raise RuntimeError('Invalid batch ID')
    token = os.environ.get('GH_TOKEN', '').strip()
    if not token:
        raise RuntimeError('Missing GH_TOKEN (Actions: write, Contents: read)')

    local = root / 'pipeline' / 'stage7_batches' / (batch_id + '.json')
    if not local.is_file():
        raise RuntimeError('Immutable batch marker missing')
    data = json.loads(local.read_text(encoding='utf-8'))
    if (data.get('batch_id') != batch_id or data.get('batch_name') != batch
            or data.get('mode') != 'TEST_ONLY_NO_PROMOTION'
            or data.get('production_approved') is not False):
        raise RuntimeError('TEST ONLY batch marker does not match requested batch')

    # Require the exact marker on remote main before dispatching.
    from urllib.parse import quote
    _, remote = github_json('GET', '/contents/pipeline/stage7_batches/' +
                            quote(batch_id + '.json') + '?ref=main', token)
    if remote.get('encoding') != 'base64':
        raise RuntimeError('Remote batch marker is unavailable')
    remote_data = json.loads(base64.b64decode(remote['content']).decode('utf-8'))
    if remote_data != data:
        raise RuntimeError('Local and GitHub batch markers differ; sync before dispatch')

    # Avoid dispatch while another Stage 7 TEST ONLY run is active.
    for status in ('queued', 'in_progress'):
        _, runs = github_json('GET', '/actions/workflows/' + WORKFLOW +
                              '/runs?branch=main&status=' + status + '&per_page=20', token)
        if runs.get('workflow_runs'):
            raise RuntimeError('Stage 7 TEST ONLY workflow has an active GitHub run')

    code, _ = github_json('POST', '/actions/workflows/' + WORKFLOW + '/dispatches',
                          token, {'ref': 'main', 'inputs': {'batch_id': batch_id}})
    if code != 204:
        raise RuntimeError('GitHub did not accept workflow dispatch')
    url = 'https://github.com/' + GITHUB_REPO + '/actions/workflows/' + WORKFLOW
    out.write('STAGE7_TEST_ONLY_DISPATCH_ACCEPTED: ' + batch_id + '\n')
    out.write('GitHub workflow: ' + url + '\n')
    return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True)
    ap.add_argument('--job-id', required=True)
    ap.add_argument('--mode', choices=('push', 'dispatch'), default='push')
    ap.add_argument('--batch-id', default='')
    args = ap.parse_args()
    if not re.fullmatch(r'batch_[0-9]{3,6}', args.batch):
        ap.error('Invalid batch name')
    if not re.fullmatch(r'[0-9a-f]{24}', args.job_id):
        ap.error('Invalid job id')

    root = Path(__file__).resolve().parent.parent
    directory = job_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = directory / 'active.lock'
    status = directory / f'{args.job_id}.json'
    log = directory / f'{args.job_id}.log'
    fd = None
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, f'{args.job_id}\n'.encode('ascii'))
        os.close(fd)
        fd = None
    except FileExistsError:
        write_status(status, state='blocked', batch=args.batch,
                     error='Another Stage 7 control job is running (active.lock).', updated_at=now())
        return 2
    try:
        write_status(status, state='running', batch=args.batch, started_at=now(), updated_at=now())
        with log.open('w', encoding='utf-8', errors='replace') as out:
            out.write(f'STAGE7_CONTROL_BATCH: {args.batch}\n')
            out.flush()
            if args.mode == 'dispatch':
                url = dispatch_test_only(root, args.batch, args.batch_id, out)
                write_status(status, state='dispatched', batch=args.batch,
                             batch_id=args.batch_id, updated_at=now(), workflow_url=url,
                             log_file=str(log))
                return 0
            proc = subprocess.run(
                [sys.executable, '-u', str(root / 'pipeline' / 'stage7_auto_trigger.py'),
                 '--once', '--batch', args.batch],
                cwd=root, stdout=out, stderr=subprocess.STDOUT, check=False,
            )
        content = log.read_text(encoding='utf-8', errors='replace')
        if proc.returncode == 0 and 'STAGE7_AUTO_TRIGGER_PUSH_PASS:' in content:
            state = 'submitted'
        elif proc.returncode == 0 and 'STAGE7_BATCH_ALREADY_SUBMITTED:' in content:
            state = 'already_submitted'
        else:
            state = 'failed'
        write_status(status, state=state, batch=args.batch, exit_code=proc.returncode,
                     updated_at=now(), log_file=str(log))
        return proc.returncode
    except Exception as exc:
        write_status(status, state='failed', batch=args.batch,
                     error=f'{type(exc).__name__}: {exc}', updated_at=now())
        return 1
    finally:
        if fd is not None:
            os.close(fd)
        try:
            if lock.read_text(encoding='ascii').strip() == args.job_id:
                lock.unlink()
        except OSError:
            pass


if __name__ == '__main__':
    sys.exit(main())
