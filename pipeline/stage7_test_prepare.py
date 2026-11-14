"""Prepare a TEST-ONLY PKLot train/valid batch. Never interprets PKLot as online data."""
import hashlib
import json
import math
import os
from io import BytesIO
from pathlib import Path
import zipfile

from PIL import Image
from stage7_prepare import CHECKPOINTS, sha256, validation_samples


def crop_png(source, bbox):
    x, y, w, h = bbox
    with Image.open(source) as im:
        im = im.convert('RGB')
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right, bottom = min(im.width, math.ceil(x + w)), min(im.height, math.ceil(y + h))
        if right <= left or bottom <= top:
            raise ValueError('Invalid crop')
        buffer = BytesIO()
        im.crop((left, top, right, bottom)).save(buffer, format='PNG')
        return buffer.getvalue()


def main():
    if not os.environ.get('GITHUB_RUN_ID') or not os.environ.get('GITHUB_SHA'):
        raise ValueError('Only run from GitHub Actions')
    dest = Path('.stage7_test_dataset')
    if dest.exists():
        raise RuntimeError('Delete prior .stage7_test_dataset before retrying')
    dest.mkdir()
    dataset_name = f"smart-parking-stage7-test-r{os.environ['GITHUB_RUN_ID']}-a{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    manifest = {
        'stage': 'stage7-test-only', 'origin': 'pklot_train_test_only',
        'test_only': True, 'production_approved': False,
        'github_run_id': os.environ['GITHUB_RUN_ID'], 'git_commit': os.environ['GITHUB_SHA'],
        'dataset_name': dataset_name, 'approved_dvc_sha256': 'NOT_APPLICABLE_TEST_ONLY',
        'data_dvc_sha256': sha256('data.dvc'),
        'train': [], 'valid': [], 'checkpoints': {},
        'train_source': 'PKLot data/train', 'validation_source': 'PKLot data/valid',
        'warning': 'Technical pipeline test only; NOT new camera data or independent final evaluation',
    }
    rows_train = validation_samples(Path('data/train'), 32)
    rows_valid = validation_samples(Path('data/valid'), 128)
    train_frames = {str(path.name) for _, path, _, _ in rows_train}
    valid_frames = {str(path.name) for _, path, _, _ in rows_valid}
    if train_frames & valid_frames:
        raise ValueError('Identical source filenames between train and validation')
    with zipfile.ZipFile(dest / 'payload.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for split, records in (('train', rows_train), ('valid', rows_valid)):
            for i, (label, src, bbox, ann_id) in enumerate(records):
                raw = crop_png(src, bbox)
                name = f'{split}/{i:05d}.png'
                archive.writestr(name, raw)
                manifest[split].append({
                    'file': name, 'label': label,
                    'sha256': hashlib.sha256(raw).hexdigest(),
                    'source_split': split, 'source_frame': src.name, 'annotation_id': ann_id,
                })
        for model, cp in CHECKPOINTS.items():
            path = Path(cp)
            if not path.is_file():
                raise FileNotFoundError(cp)
            stored = f'checkpoint/{path.name}'
            archive.write(path, stored)
            manifest['checkpoints'][model] = {'file': stored, 'sha256': sha256(path)}
    with zipfile.ZipFile(dest / 'payload.zip') as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 64 + 256 + 3
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (dest / 'dataset-metadata.json').write_text(json.dumps({
        'title': dataset_name, 'id': f'nguynngthibo/{dataset_name}',
        'licenses': [{'name': 'unknown'}],
    }, indent=2), encoding='utf-8')
    print('STAGE7_TEST_ONLY_DATA_PREP_PASS: 64 PKLot train crops, 256 PKLot valid crops')
    print('STAGE7_TEST_ONLY_NO_ONLINE_DATA_PASS')


if __name__ == '__main__':
    main()
