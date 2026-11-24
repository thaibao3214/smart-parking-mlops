"""Conservative Smart Parking MLOps cleanup.
Default: preview only. --apply: move untracked .pyc and .bak files to a dated
archive OUTSIDE the project. --include-zips adds top-level GitHub artifact ZIPs.
Never renames active sources, modifies DVC data, or deletes files.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess


def tracked_paths(root: Path) -> set[str] | None:
    try:
        r = subprocess.run(["git", "ls-files", "-z"], cwd=root, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, check=True, timeout=20)
        return {p.decode('utf-8', 'surrogateescape').replace('\\', '/') for p in r.stdout.split(b'\0') if p}
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description='Preview/archive safe generated files; no deletion.')
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--apply', action='store_true', help='Move candidates outside project (default: preview)')
    ap.add_argument('--include-zips', action='store_true', help='Also archive top-level smart-parking stage7 artifact ZIPs')
    args = ap.parse_args()
    root = args.root.resolve()
    if not (root / 'dashboard' / 'app.py').is_file() or not (root / '.git').exists():
        ap.error('Root must be the Git repository containing dashboard/app.py')
    tracked = tracked_paths(root)
    if tracked is None:
        ap.error('Cannot read git tracked files; no cleanup performed')
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    archive = root.parent / f'{root.name}_cleanup_archive_{timestamp}'
    chosen = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {'.git', '.dvc', '.venv', 'venv', 'node_modules'}]
        for name in files:
            src = Path(base) / name
            rel = src.relative_to(root).as_posix()
            if rel in tracked or src.is_symlink():
                continue
            category = None
            if name.endswith(('.pyc', '.bak')):
                category = 'generated_or_backup'
            elif (args.include_zips and src.parent == root and
                  name.startswith('smart-parking-stage7-TEST-ONLY-') and name.endswith('.zip')):
                category = 'old_stage7_artifact_zip'
            if category:
                chosen.append((src, rel, category, src.stat().st_size))
    chosen.sort(key=lambda row: row[1])
    total = sum(c[3] for c in chosen)
    print(f'Project: {root}')
    print(f'Mode: {"ARCHIVE" if args.apply else "PREVIEW ONLY"}')
    print(f'Candidates: {len(chosen)}; total: {total / 1024**2:.2f} MiB')
    print(f'Archive destination: {archive}')
    for _, rel, category, size in chosen:
        print(f'  [{category}] {rel} ({size / 1024**2:.2f} MiB)')
    if not args.apply:
        print('\nNo files moved. Use --apply after reviewing the list.')
        return
    if not chosen:
        print('Nothing to move')
        return
    archive.mkdir(parents=True, exist_ok=False)
    with (archive / 'manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['relative_path', 'category', 'bytes'])
        for src, rel, category, size in chosen:
            # Revalidate just before any move.
            if rel in tracked_paths(root) or not src.is_file() or src.is_symlink():
                print(f'SKIPPED changed/tracked: {rel}')
                continue
            dest = archive / Path(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            writer.writerow([rel, category, size])
    print(f'Archived files at: {archive}')
    print('Restore by moving the archived relative paths back to the project root.')

if __name__ == '__main__':
    main()
