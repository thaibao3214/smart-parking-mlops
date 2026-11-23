"""Audit exactly two rejected frames; never delete original evidence."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_exclusions(sim, confirmed, manifest):
    path = confirmed / 'excluded_frames.json'
    if not path.exists():
        return set()
    doc = json.loads(path.read_text(encoding='utf-8'))
    if (set(doc) != {'schema_version', 'batch_name', 'excluded_frames', 'review_method', 'test_only'}
            or doc['schema_version'] != 1 or doc['batch_name'] != sim.name
            or doc['test_only'] is not True or doc['review_method'] != 'manual_quality_review'):
        raise ValueError('Invalid exclusion record')
    items = doc['excluded_frames']
    if not isinstance(items, list) or not (1 <= len(items) <= 2):
        raise ValueError('Expected one or two manually excluded frames')
    mapping = {(str(r['image_id']), r['file_name']) for r in manifest['images']}
    excluded = set()
    for item in items:
        if set(item) != {'source_image_id', 'file_name', 'reason'}:
            raise ValueError('Invalid excluded frame metadata')
        pair = (str(item['source_image_id']), item['file_name'])
        if pair not in mapping or len(item['reason'].strip()) < 8:
            raise ValueError('Excluded frame not in manifest / missing reason')
        excluded.add(pair[0])
    if len(excluded) != len(items):
        raise ValueError('Duplicated excluded frame')
    return excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--frame', action='append', required=True,
                        help='Specify twice, e.g. --frame frame_009.jpg --frame frame_015.jpg')
    parser.add_argument('--reason', required=True)
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'batch_[0-9]{3,6}', args.batch):
        parser.error('Invalid batch name')
    if not (1 <= len(args.frame) <= 2) or len(set(args.frame)) != len(args.frame) or any(
        not re.fullmatch(r'frame_[0-9]{3,6}\.(?:jpg|jpeg|png)', f, flags=re.I)
        for f in args.frame
    ):
        parser.error('Provide one or two distinct frame filenames')
    if len(args.reason.strip()) < 8:
        parser.error('Provide a meaningful exclusion reason')
    sim = ROOT / 'simulation_batches' / args.batch
    confirmed = ROOT / 'confirmed_batches' / args.batch
    manifest = json.loads((sim / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('origin') != 'pklot_train_simulation'
            or manifest.get('test_only') is not True
            or manifest.get('batch_name') != args.batch):
        raise ValueError('Invalid TEST_ONLY simulation manifest')
    by_name = {item['file_name']: item for item in manifest['images']}
    if any(f not in by_name for f in args.frame):
        raise ValueError('One or more excluded frames not found in batch manifest')
    target = confirmed / 'excluded_frames.json'
    if target.exists():
        raise FileExistsError(f'Existing exclusion record must not be overwritten: {target}')
    doc = {
        'schema_version': 1, 'batch_name': args.batch,
        'excluded_frames': [
            {'source_image_id': by_name[f]['image_id'], 'file_name': f,
             'reason': args.reason.strip()} for f in args.frame
        ],
        'review_method': 'manual_quality_review', 'test_only': True,
    }
    print('EXCLUSIONS_PREVIEW:', json.dumps(doc, ensure_ascii=False))
    if not args.confirm:
        print('DRY_RUN_ONLY: add --confirm after reviewing exclusions')
        return
    confirmed.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as out:
        json.dump(doc, out, indent=2, ensure_ascii=False)
        out.write('\n')
    print('STAGE7_EXCLUSIONS_PASS:', len(args.frame))
    print('Original frames and any manual labels remain intact')


if __name__ == '__main__':
    main()
