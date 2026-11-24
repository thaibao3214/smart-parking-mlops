"""Read-only monitoring for TEST_ONLY GitHub Actions runs.

A run is associated with a batch ONLY when its job log contains the immutable
batch marker emitted by the workflow's 'Resolve immutable batch marker' step.
"""
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone

OWNER_REPO = 'thaibao3214/smart-parking-mlops'
WORKFLOW = 'stage7_test_only.yml'
API = 'https://api.github.com/repos/' + OWNER_REPO
MARKER = re.compile(r'STAGE7_RESOLVED_IMMUTABLE_BATCH:\s+(batch_[0-9]{3,6})\s+([0-9a-f]{20})')


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            original = urllib.parse.urlparse(req.full_url).netloc
            destination = urllib.parse.urlparse(newurl).netloc
            if original != destination:
                # Do not forward a GitHub PAT to artifact/log download CDNs.
                redirected.remove_header('Authorization')
                redirected.headers.pop('Authorization', None)
                redirected.unredirected_hdrs.pop('Authorization', None)
        return redirected


def _request(url, token, max_bytes=8_000_000):
    if not url.startswith(('https://api.github.com/', 'https://github.com/')):
        # urllib follows GitHub's signed redirects, potentially to githubusercontent.
        # Initial URLs must always belong to the trusted API.
        raise RuntimeError('Untrusted GitHub URL')
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28',
               'User-Agent': 'smart-parking-stage7-readonly-monitor'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.build_opener(_SafeRedirect()).open(req, timeout=35) as res:
            data = res.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise RuntimeError('GitHub result exceeds safe size limit')
            return data
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'GitHub monitoring HTTP {e.code}') from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError('Không kết nối được GitHub API để theo dõi.') from None


def _json(path, token):
    return json.loads(_request(API + path, token).decode('utf-8'))


def _time(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _job_log_matches(run_id, batch, batch_id, token):
    jobs = _json(f'/actions/runs/{int(run_id)}/jobs?per_page=30', token).get('jobs', [])
    for job in jobs:
        if job.get('name') != 'test-only':
            continue
        # GitHub's job logs endpoint returns text (often via a 302 redirect).
        body = _request(API + f'/actions/jobs/{int(job["id"])}/logs', token,
                        max_bytes=12_000_000)
        # API logs can be plain text or a ZIP, depending on endpoint behavior.
        if body.startswith(b'PK\x03\x04'):
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                text = '\n'.join(archive.read(n).decode('utf-8', 'replace')
                                 for n in archive.namelist() if n.endswith(('.txt', '.log')))
        else:
            text = body.decode('utf-8', 'replace')
        return any(b == batch and ident == batch_id for b, ident in MARKER.findall(text))
    return False


def matching_run(batch, batch_id, dispatched_at, token):
    """Find a confirmed matching run, never guess from run order or time alone.

    Caller should trigger only on a deliberate 'Check GitHub' button, to avoid
    rate limits and surprise network traffic on every Streamlit rerender.
    """
    if not re.fullmatch(r'batch_[0-9]{3,6}', str(batch)) or not re.fullmatch(r'[0-9a-f]{20}', str(batch_id)):
        raise ValueError('Invalid batch metadata')
    since = _time(dispatched_at)
    if since is None:
        raise ValueError('Missing dispatch timestamp')
    # GitHub can create a run slightly before we write local status.
    since -= timedelta(minutes=3)
    data = _json('/actions/workflows/' + WORKFLOW + '/runs?event=workflow_dispatch&branch=main&per_page=35', token)
    candidates = [r for r in data.get('workflow_runs', [])
                  if _time(r.get('created_at')) and _time(r['created_at']) >= since]
    # Bound checks. The commit/run timestamp is never evidence enough alone.
    for run in candidates[:12]:
        try:
            if _job_log_matches(run['id'], batch, batch_id, token):
                return {'id': run['id'], 'url': run.get('html_url'),
                        'status': run.get('status'), 'conclusion': run.get('conclusion'),
                        'created_at': run.get('created_at'),
                        'head_sha': run.get('head_sha')}
        except RuntimeError:
            # In-progress jobs might not expose logs yet; do not misattribute.
            continue
    return None


def summary_from_artifact(run_id, token):
    """Read only a small, dedicated Stage 7 report artifact.

    Never download model checkpoints via Streamlit. Older runs without a
    lightweight report remain viewable in their GitHub Actions job summary.
    """
    run_id = int(run_id)
    data = _json(f'/actions/runs/{run_id}/artifacts?per_page=100', token)
    expected = f'smart-parking-stage7-REPORT-{run_id}-'
    matches = [a for a in data.get('artifacts', [])
               if str(a.get('name', '')).startswith(expected) and not a.get('expired')]
    if not matches:
        raise RuntimeError(
            'Run cũ chưa có artifact REPORT nhỏ. Mở GitHub Actions → Summary '
            'để xem kết quả; không cần chạy retraining lại. '
            'Các run sau khi cập nhật workflow sẽ có artifact REPORT riêng.')
    artifact = max(matches, key=lambda a: a.get('created_at', ''))
    if not (0 < int(artifact.get('size_in_bytes', 0)) <= 2_000_000):
        raise RuntimeError('Artifact REPORT có kích thước bất thường (>2MB). Từ chối tải.')
    binary = _request(API + f'/actions/artifacts/{int(artifact["id"])}/zip', token,
                      max_bytes=2_000_000)
    with zipfile.ZipFile(io.BytesIO(binary)) as archive:
        if len(archive.infolist()) > 10:
            raise RuntimeError('Artifact REPORT có quá nhiều file.')
        md = next((n for n in archive.namelist() if n.endswith('stage7_summary.md')), None)
        report = next((n for n in archive.namelist() if n.endswith('stage7_test_report.json')), None)
        if not md and not report:
            raise RuntimeError('Artifact REPORT không chứa báo cáo cần thiết.')
        for n in (md, report):
            if n and archive.getinfo(n).file_size > 1_000_000:
                raise RuntimeError('File báo cáo quá lớn.')
        text = archive.read(md).decode('utf-8', 'replace') if md else ''
        parsed = json.loads(archive.read(report).decode('utf-8')) if report else None
    return text, parsed
