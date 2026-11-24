"""Auditable TEST_ONLY frame exclusions for the Streamlit reviewer.

Does not use predicted labels as ground truth, delete images, or promote models.
"""
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MAX_EXCLUSIONS = 2  # Matches the current Stage7 quality policy.


def _validate_batch(batch, manifest):
    if not re.fullmatch(r'batch_[0-9]{3,6}', batch):
        raise ValueError('Batch name không hợp lệ')
    if (manifest.get('batch_name') != batch
            or manifest.get('origin') != 'pklot_train_simulation'
            or manifest.get('test_only') is not True):
        raise ValueError('Manifest không phải PKLot train TEST_ONLY của batch này')


def read_exclusion_ids(stream, confirmed, manifest):
    """Read and validate 0..2 exclusions; no mutation."""
    batch = stream.name
    _validate_batch(batch, manifest)
    path = confirmed / 'excluded_frames.json'
    if not path.is_file():
        return set()
    doc = json.loads(path.read_text(encoding='utf-8'))
    if (set(doc) != {'schema_version', 'batch_name', 'excluded_frames', 'review_method', 'test_only'}
            or doc['schema_version'] != 1 or doc['batch_name'] != batch
            or doc['test_only'] is not True or doc['review_method'] != 'manual_quality_review'):
        raise ValueError('Exclusion record không hợp lệ')
    items = doc['excluded_frames']
    if not isinstance(items, list) or not (1 <= len(items) <= MAX_EXCLUSIONS):
        raise ValueError('Số frame loại trừ không hợp lệ')
    allowed = {(str(f['image_id']), f['file_name']) for f in manifest['images']}
    result = set()
    for item in items:
        if (set(item) != {'source_image_id', 'file_name', 'reason'}
                or (str(item['source_image_id']), item['file_name']) not in allowed
                or not isinstance(item['reason'], str) or len(item['reason'].strip()) < 8):
            raise ValueError('Thông tin frame loại trừ không hợp lệ')
        result.add(str(item['source_image_id']))
    if len(result) != len(items):
        raise ValueError('Frame loại trừ bị trùng')
    return result


def exclude_frame(stream, confirmed, manifest, item, reason):
    """Append a reviewed exclusion (max 2); lock and atomic file replacement."""
    batch = stream.name
    _validate_batch(batch, manifest)
    if not isinstance(reason, str) or len(reason.strip()) < 8:
        raise ValueError('Vui lòng nhập lý do loại trừ có ý nghĩa')
    allowed = {(str(f['image_id']), f['file_name']) for f in manifest['images']}
    image_id, filename = str(item['image_id']), item['file_name']
    if (image_id, filename) not in allowed:
        raise ValueError('Frame không thuộc batch')
    if Path(filename).name != filename:
        raise ValueError('Tên frame không hợp lệ')
    frame_key = f'{image_id}_{Path(filename).stem}'
    if (confirmed / 'frames' / frame_key).exists():
        raise ValueError('Frame đã được xác nhận. Không tự động thay đổi nhãn đã lưu.')
    confirmed.mkdir(parents=True, exist_ok=True)
    lock = confirmed / '.exclusion_review.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError('Có một thao tác loại trừ khác đang diễn ra; kiểm tra lại trước khi thử lại') from exc
    os.close(fd)
    tmp = None
    try:
        existing_ids = read_exclusion_ids(stream, confirmed, manifest)
        if image_id in existing_ids:
            raise ValueError('Frame đã được loại trừ trước đó')
        if len(existing_ids) >= MAX_EXCLUSIONS:
            raise ValueError('Batch đã đạt giới hạn 2 frame loại trừ; cần kiểm tra chính sách chất lượng')
        target = confirmed / 'excluded_frames.json'
        if target.exists():
            doc = json.loads(target.read_text(encoding='utf-8'))
        else:
            doc = {'schema_version': 1, 'batch_name': batch, 'excluded_frames': [],
                   'review_method': 'manual_quality_review', 'test_only': True}
        doc['excluded_frames'].append({'source_image_id': item['image_id'],
                                       'file_name': filename, 'reason': reason.strip()})
        with tempfile.NamedTemporaryFile('w', dir=confirmed, prefix='.exclusion_',
                                         suffix='.json', delete=False, encoding='utf-8') as handle:
            tmp = Path(handle.name)
            json.dump(doc, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        tmp = None
        # History is supplemental; excluded_frames.json remains the canonical contract.
        with (confirmed / 'exclusion_events.jsonl').open('a', encoding='utf-8') as audit:
            audit.write(json.dumps({'event': 'exclude_frame', 'batch': batch,
                                    'image_id': image_id, 'file_name': filename,
                                    'reason': reason.strip(),
                                    'at_utc': datetime.now(timezone.utc).isoformat()},
                                   ensure_ascii=False) + '\n')
        return len(doc['excluded_frames'])
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
