#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only audit for a Smart Parking MLOps project.

Run from the project root:
    python audit_full_project.py
    python audit_full_project.py --root C:\\path\\to\\project

Outputs: project_audit/full_audit_report.json and full_audit_report.md
No imports of project modules, no training, no deployment, no network calls.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

SKIP = {'.git', '.dvc', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache', '.mypy_cache'}
PY_EXT = '.py'
MAX_PY_BYTES = 2_000_000
MAX_TEXT_BYTES = 600_000
MAX_SCAN_FILES = 25000
MAX_JSON_BYTES = 18_000_000
SENSITIVE = re.compile(r'(?i)(token|password|secret|api[_-]?key|credential|private[_-]?key)')


def readable(p: Path, limit: int = MAX_TEXT_BYTES) -> str:
    try:
        if p.stat().st_size > limit:
            return ''
        return p.read_text(encoding='utf-8-sig', errors='replace')
    except (OSError, ValueError):
        return ''


def rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def walk(root: Path):
    n = 0
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith('.tox')]
        for name in files:
            n += 1
            if n > MAX_SCAN_FILES:
                return
            yield Path(base) / name


def first_existing(root: Path, patterns: list[str]) -> list[Path]:
    found = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    return [p for p in found if p.is_file()]


def check_python(root: Path, py_files: list[Path]):
    errors, parsed, features = [], 0, {}
    for p in py_files:
        try:
            if p.stat().st_size > MAX_PY_BYTES:
                errors.append({'file': rel(root, p), 'error': 'Python file too large to inspect'})
                continue
            code = readable(p, MAX_PY_BYTES)
            tree = ast.parse(code, filename=str(p))
            parsed += 1
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        names.add(node.func.attr)
            features[rel(root, p)] = sorted(names)
        except (SyntaxError, ValueError, OSError) as e:
            errors.append({'file': rel(root, p), 'error': str(e).split('\n')[0][:200]})
    return {'parsed': parsed, 'total': len(py_files), 'syntax_errors': errors}, features


def probe_git(root: Path):
    if not (root / '.git').exists():
        return {'status': 'not_checked', 'note': 'No .git directory; git metadata unavailable'}
    try:
        cmd = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True,
                             text=True, timeout=8, check=False)
        if cmd.returncode:
            return {'status': 'unknown', 'note': 'git status returned an error'}
        count = sum(1 for x in cmd.stdout.splitlines() if x.strip())
        return {'status': 'clean' if count == 0 else 'modified', 'changed_files': count}
    except (OSError, subprocess.TimeoutExpired):
        return {'status': 'unknown', 'note': 'Git executable unavailable or timed out'}


def inspect_coco(root: Path, files: list[Path]):
    candidates = [p for p in files if p.suffix.lower() == '.json' and
                  ('annot' in p.name.lower() or 'coco' in p.name.lower() or
                   'instances' in p.name.lower()) and
                  not any(part in {'project_audit', 'stage7_batches'} for part in p.parts)]
    output = []
    for p in candidates[:12]:
        try:
            if p.stat().st_size > MAX_JSON_BYTES:
                output.append({'file': rel(root, p), 'status': 'skipped_large'})
                continue
            data = json.loads(readable(p, MAX_JSON_BYTES))
            if not isinstance(data, dict) or not isinstance(data.get('images'), list) or not isinstance(data.get('annotations'), list):
                continue
            cats = {c.get('id'): c.get('name') for c in data.get('categories', []) if isinstance(c, dict)}
            counts = Counter(c.get('category_id') for c in data['annotations'] if isinstance(c, dict))
            output.append({'file': rel(root, p), 'status': 'readable',
                           'images': len(data['images']), 'annotations': len(data['annotations']),
                           'categories': {str(k): v for k, v in cats.items()},
                           'annotation_counts': {str(k): v for k, v in counts.items()}})
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            output.append({'file': rel(root, p), 'status': 'unreadable'})
    return output


def get_stage_result(root: Path, files: list[Path]):
    # Only report text evidence and references, never claim verified performance based on file names.
    hits = []
    pattern = re.compile(r'(?i)(macro.?f1|production.{0,35}unchanged|candidate.{0,35}(fail|pass)|gate.{0,15}(fail|pass))')
    for p in files:
        if p.suffix.lower() not in {'.md', '.json', '.txt', '.csv'}:
            continue
        if not ('stage7' in str(p).lower() or 'gate' in p.name.lower() or 'metrics' in p.name.lower()):
            continue
        content = readable(p, 250_000)
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if pattern.search(line):
                hits.append({'file': rel(root, p), 'line': i + 1,
                             'excerpt': line.strip()[:170]})
                if len(hits) >= 45:
                    return hits
    return hits


def audit(root: Path):
    files = list(walk(root))
    py_files = [p for p in files if p.suffix.lower() == PY_EXT]
    syntax, symbols = check_python(root, py_files)

    sections = []
    def record(name, required=(), any_of=(), markers=(), note=''):
        req = [(path, (root / path).exists()) for path in required]
        groups = [(desc, bool(first_existing(root, pats))) for desc, pats in any_of]
        signals = []
        for desc, paths, terms in markers:
            matched = []
            for path in paths:
                p = root / path
                if p.is_file():
                    src = readable(p).lower()
                    for term in terms:
                        if term.lower() in src:
                            matched.append(path)
                            break
            signals.append((desc, bool(matched), matched))
        total = len(req) + len(groups) + len(signals)
        hits = sum(ok for _, ok in req) + sum(ok for _, ok in groups) + sum(ok for _, ok, _ in signals)
        # Readiness means evidence presence only; NOT test pass nor deployment certification.
        if total == 0 or hits == 0:
            status = 'CHUA_THAY_BANG_CHUNG'
        elif hits == total:
            status = 'CO_THANH_PHAN'
        else:
            status = 'MOT_PHAN'
        sections.append({'name': name, 'status': status, 'signals_met': hits,
                         'signals_total': total,
                         'checks': [{'item': p, 'found': ok} for p, ok in req] +
                                   [{'item': desc, 'found': ok} for desc, ok in groups] +
                                   [{'item': desc, 'found': ok, 'files': matched} for desc, ok, matched in signals],
                         'note': note})

    record('01 - Dataset / COCO',
           required=['data'],
           any_of=[('COCO annotations', ['data/**/*_annotations*.json', 'data/**/instances*.json', 'data/**/annotations*.json', 'kaggle_dataset/**/*.json', 'data/**/*.json']),
                   ('Dataset verification script', ['check_dataset.py', 'validate_online_data.py'])],
           note='Presence only; dataset quality and label accuracy require manual validation.')
    record('02 - Data versioning (DVC)',
           required=['.dvc', 'data.dvc'],
           any_of=[('Other DVC-tracked datasets', ['*.dvc'])],
           note='Does not execute dvc pull or verify remote.')
    record('03 - Model implementations',
           required=['cnn_model', 'mobilenet_model', 'resnet_model'],
           any_of=[('Model-related Python files', ['cnn_model/**/*.py', 'mobilenet_model/**/*.py', 'resnet_model/**/*.py'])])
    record('04 - MLflow / Registry / DagsHub',
           any_of=[('MLflow setup or registration', ['migrate_mlflow.py', 'register_v1_reliable.py', 'upload_mobilenet.py']),
                   ('MLflow or registry tests', ['test_dagshub.py', 'test_registry.py'])],
           note='No remote authentication/MLflow API calls; production alias not verified live.')
    record('05 - API / Inference',
           required=['api'],
           any_of=[('API Python code', ['api/**/*.py'])],
           note='No request sent to API; endpoints and latency are not verified.')
    record('06 - Streamlit dashboard',
           required=['dashboard/app.py'],
           any_of=[('Dashboard additional modules', ['dashboard/mlops_monitoring.py', 'dashboard/online_stream.py'])],
           markers=[('Streamlit usage', ['dashboard/app.py'], ['streamlit', 'st.'])],
           note='Static evidence only; dashboard is not launched.')
    record('07 - Online streaming / feedback',
           required=['dashboard/online_stream.py'],
           any_of=[('Batch data folders', ['confirmed_batches/*', 'simulation_batches/*'])],
           note='Confirmed batches may be simulated; do not assume real camera data.')
    record('08 - Retraining',
           required=['retraining/retrain.py', 'retraining/evaluate.py'],
           any_of=[('Retraining readiness checker', ['check_retraining_readiness.py'])],
           note='No retraining triggered; readiness and execution are distinct.')
    record('09 - Stage 5 and 6 evaluation',
           required=['pipeline/stage5_evaluate.py', 'pipeline/stage6_evaluate.py'],
           any_of=[('Stage 5/6 docs', ['README_STAGE5.md', 'README_STAGE6.md'])])
    record('10 - Stage 7 continuous training / gate',
           required=['pipeline/stage7_prepare.py', 'pipeline/stage7_gate.py', 'pipeline/stage7_promote.py'],
           any_of=[('Stage 7 automatic trigger', ['pipeline/stage7_auto_trigger.py']),
                   ('Stage 7 docs', ['README_STAGE7.md', 'README_STAGE7_AUTO.md'])],
           note='Gate files existing does not imply gate passed. Promotion must remain human-approved.')
    record('11 - Stage 7 UI controls and QA',
           required=['dashboard/stage7_control.py', 'dashboard/stage7_control_worker.py'],
           any_of=[('Human review actions', ['dashboard/stage7_review_actions.py']),
                   ('QA scripts', ['pipeline/stage7_auto_register_qa.py', 'pipeline/stage7_register_qa_v2.py'])])
    record('12 - CI/CD GitHub Actions',
           required=['.github'],
           any_of=[('Workflow definitions', ['.github/workflows/*.yml', '.github/workflows/*.yaml'])],
           note='Workflow status in GitHub is NOT checked.')
    record('13 - Tests and documentation',
           required=['requirements.txt'],
           any_of=[('Test Python files', ['test*.py', '**/test_*.py']),
                   ('README docs', ['README*.md', 'docs/**/*.md'])],
           note='Tests not executed.')

    stage7_ui = root / 'dashboard' / 'stage7_control.py'
    app = root / 'dashboard' / 'app.py'
    ui_text = '\n'.join(readable(p).lower() for p in [stage7_ui, app, root / 'dashboard/stage7_control_worker.py'])
    retrain_text = readable(root / 'retraining' / 'retrain.py').lower()
    trigger_text = readable(root / 'pipeline' / 'stage7_auto_trigger.py').lower()
    integrations = {
        'dashboard_retraining_related_words': bool(re.search(r'retrain|stage7_auto_trigger|start.training', ui_text)),
        'dashboard_launches_process_or_workflow': bool(re.search(r'subprocess|workflow_dispatch|github\.com|stage7_control_worker', ui_text)),
        'retrain_script_defines_training_operations': bool(re.search(r'(fit\s*\(|train\s*\(|optimizer|backward\s*\()', retrain_text)),
        'auto_trigger_contains_dispatch_related_code': bool(re.search(r'workflow_dispatch|subprocess|requests\.|httpx\.', trigger_text)),
    }
    # No inference that end-to-end invocation is working.
    missing = [f"{s['name']}: {c['item']}" for s in sections for c in s['checks'] if not c['found']]
    summary = Counter(s['status'] for s in sections)
    return {
        'audit_time_local': dt.datetime.now().astimezone().isoformat(timespec='seconds'),
        'root': str(root),
        'scope': 'Read-only static evidence audit, not a functional end-to-end test',
        'counts': {'scanned_files': len(files), 'python_files': len(py_files),
                   'directories_top_level': len([p for p in root.iterdir() if p.is_dir()])},
        'summary': dict(summary), 'sections': sections, 'missing_evidence': missing,
        'python_syntax': syntax, 'coco_examples': inspect_coco(root, files),
        'integration_signals': integrations,
        'stage7_report_excerpts': get_stage_result(root, files),
        'git': probe_git(root),
        'limits': ['File existence is not proof that a feature runs.',
                   'No network/MLflow/DagsHub/GitHub checks were performed.',
                   'No tests, retraining, inference, DVC actions or production promotion were run.',
                   'No secret contents are printed.'],
    }


def report_md(result: dict) -> str:
    lines = ['# Smart Parking MLOps — Full Project Audit', '',
             f"- Time: {result['audit_time_local']}",
             f"- Root: `{result['root']}`",
             f"- Mode: {result['scope']}",
             f"- Files scanned: {result['counts']['scanned_files']}; Python files: {result['counts']['python_files']}", '',
             '## Status by workstream', '',
             '| Area | Status | Evidence |', '|---|---|---|']
    for s in result['sections']:
        lines.append(f"| {s['name']} | {s['status']} | {s['signals_met']}/{s['signals_total']} |")
    lines += ['', '**Important:** CO_THANH_PHAN = files/signals exist, NOT that training, deployment or any workflow passed.', '',
              '## Missing or incomplete signals', '']
    if result['missing_evidence']:
        lines += ['- ' + item for item in result['missing_evidence']]
    else:
        lines.append('- No expected static signals missing. Live functionality still unverified.')
    lines += ['', '## Python syntax', '',
              f"- Parsed: {result['python_syntax']['parsed']}/{result['python_syntax']['total']}",
              f"- Syntax errors: {len(result['python_syntax']['syntax_errors'])}"]
    for e in result['python_syntax']['syntax_errors'][:25]:
        lines.append(f"- `{e['file']}`: {e['error']}")
    lines += ['', '## Streamlit → retraining integration clues (NOT functional proof)', '']
    for k, v in result['integration_signals'].items():
        lines.append(f'- {k}: {"FOUND" if v else "NOT FOUND"}')
    lines += ['', '## COCO examples (sampled, metadata only)', '']
    for d in result['coco_examples']:
        lines.append(f"- `{d['file']}`: {d['status']}; images={d.get('images', 'n/a')}; annotations={d.get('annotations', 'n/a')}")
    if not result['coco_examples']:
        lines.append('- No readable COCO candidate found in scan bounds.')
    lines += ['', '## Stage 7 artifact excerpts (unverified report text)', '']
    for d in result['stage7_report_excerpts'][:25]:
        snippet = d['excerpt'].replace('|', '\\|').replace('`', "'")
        lines.append(f"- `{d['file']}:{d['line']}`: {snippet}")
    if not result['stage7_report_excerpts']:
        lines.append('- None found.')
    lines += ['', '## Git', '', f"- Status: {result['git'].get('status', 'unknown')}", '',
              '## What must still be verified manually', '',
              '- Run local unit/integration tests and inspect their pass/fail reports.',
              '- Start Streamlit and verify its buttons call the intended retraining workflow.',
              '- Verify dataset labels, train/val/test split, data leakage, and DVC remote.',
              '- Check MLflow/DagsHub live run and registry status.',
              '- Check actual GitHub Actions run results, Stage 7 gate metrics and human approval.',
              '- Confirm Production alias stays unchanged on FAIL.', '',
              '## Safety and limitations', '']
    lines += ['- ' + item for item in result['limits']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Read-only Smart Parking MLOps audit')
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Project root directory')
    parser.add_argument('--output', type=Path, default=None, help='Output directory (default: root/project_audit)')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Project directory does not exist: {root}')
    output = (args.output.expanduser().resolve() if args.output else root / 'project_audit')
    result = audit(root)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / 'full_audit_report.json'
    md_path = output / 'full_audit_report.md'
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    md_path.write_text(report_md(result), encoding='utf-8')
    print('\n=== SMART PARKING MLOPS PROJECT AUDIT ===')
    print('Audit mode: static/read-only (no training or deployment)')
    for s in result['sections']:
        print(f"[{s['status']:<22}] {s['name']}  ({s['signals_met']}/{s['signals_total']})")
    print(f"\nPython syntax errors: {len(result['python_syntax']['syntax_errors'])}")
    print(f"Missing static signals: {len(result['missing_evidence'])}")
    print(f'JSON: {json_path}')
    print(f'Markdown: {md_path}')
    print('\nNOTE: Static evidence is NOT proof of a successful retraining or deployment.')


if __name__ == '__main__':
    main()
