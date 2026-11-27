# SMART_PARKING_FRIENDLY_SIMULATION_LABELS_V1
# SMART_PARKING_HIDE_BATCH_ORDINALS_V1
"""Unified camera simulator + read-only Stage 7 job summary (TEST_ONLY).

One worker does the capture, prediction and GT verification; this page only
starts and visualizes that worker. No duplicate capture pipeline in Streamlit.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

STATUS_DIR = Path.home() / '.smartparking_stage7_control'
GLOBAL_LOCK = STATUS_DIR / 'auto_batch_active.lock'
BATCH_RE = re.compile(r'batch_(\d{3,6})\Z')
ACTIVE_STATES = {'queued', 'preflight', 'creating', 'created', 'simulating',
                 'verifying', 'verified', 'publishing'}
STATE_LABELS = {
    'not_started': 'Chưa chạy', 'queued': 'Đã xếp hàng',
    'preflight': 'Kiểm tra hệ thống', 'creating': 'Tạo ảnh mô phỏng',
    'created': 'Đã tạo ảnh', 'simulating': 'Đang phát camera / dự đoán',
    'verifying': 'Kiểm tra dữ liệu', 'verified': 'Dữ liệu hợp lệ',
    'publishing': 'Đẩy DVC và GitHub', 'submitted': 'Đã gửi GitHub Actions',
    'failed': 'Đã dừng do lỗi',
}


def status_path(batch: str) -> Path:
    if not BATCH_RE.fullmatch(batch):
        raise ValueError('Invalid batch name')
    return STATUS_DIR / f'auto_{batch}.json'


def read_status(batch: str) -> dict:
    path = status_path(batch)
    if not path.is_file():
        return {}
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(record, dict) or record.get('batch') not in (batch, None):
            raise ValueError('Status does not match batch')
        return record
    except (OSError, ValueError, TypeError):
        return {'batch': batch, 'state': 'unknown', 'error': 'Không đọc được trạng thái batch.'}


def latest_status() -> tuple[str, dict]:
    candidates = []
    for path in STATUS_DIR.glob('auto_batch_*.json'):
        name = path.stem.removeprefix('auto_')
        if BATCH_RE.fullmatch(name):
            candidates.append(name)
    if not candidates:
        return 'batch_005', {}
    name = max(candidates, key=lambda value: int(value.split('_')[1]))
    return name, read_status(name)


def next_batch(root: Path) -> str:
    numbers = [5]  # Batch 005 is immutable, already used as integration evidence.
    for folder in ('simulation_batches', 'confirmed_batches'):
        for path in (root / folder).glob('batch_*'):
            match = BATCH_RE.fullmatch(path.stem)
            if match:
                numbers.append(int(match.group(1)))
    for path in STATUS_DIR.glob('auto_batch_*.json'):
        match = BATCH_RE.fullmatch(path.stem.removeprefix('auto_'))
        if match:
            numbers.append(int(match.group(1)))
    marker_dir = root / 'pipeline' / 'stage7_batches'
    for marker in marker_dir.glob('*.json'):
        try:
            record = json.loads(marker.read_text(encoding='utf-8'))
            value = record.get('batch_name', '')
            match = BATCH_RE.fullmatch(str(value))
            if match:
                numbers.append(int(match.group(1)))
        except (OSError, ValueError, TypeError):
            continue
    new_number = max(numbers) + 1
    if new_number > 999999:
        raise RuntimeError('No available batch ID')
    return f'batch_{new_number:03d}'


def _check_api(api_url: str) -> str:
    parsed = urlparse(api_url.strip())
    if (parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1'}
            or parsed.username or parsed.password or parsed.path not in {'', '/'}
            or parsed.query or parsed.fragment):
        raise ValueError('FastAPI phải dùng HTTP localhost / 127.0.0.1')
    return api_url.strip().rstrip('/')


def _git_ready(root: Path) -> None:
    files = ['dashboard/online_stream.py', 'dashboard/stage7_unified_camera_ui.py',
             'dashboard/stage7_control.py', 'pipeline/stage7_batch5_worker.py',
             'pipeline/stage7_auto_trigger.py', 'pipeline/stage7_test_prepare.py']
    for args in (['git', 'status', '--porcelain', '--', *files],):
        result = subprocess.run(args, cwd=root, text=True, capture_output=True,
                                encoding='utf-8', errors='replace', timeout=20, check=True)
        if result.stdout.strip():
            raise RuntimeError('Cần commit các file nâng cấp trước khi bắt đầu batch mới.')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                   text=True, timeout=20).strip()
    remote = subprocess.check_output(['git', 'rev-parse', 'github/main'], cwd=root,
                                     text=True, timeout=20).strip()
    if head != remote:
        raise RuntimeError('Nhánh local chưa đồng bộ github/main. Hãy push code trước.')
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'],
                                     cwd=root, text=True, timeout=20).strip()
    if staged:
        raise RuntimeError('Git đang có file staged; cần commit hoặc unstage trước.')


def start_next(root: Path, api_url: str) -> tuple[str, int]:
    """One atomic start. Never rerun submitted jobs or take over manual batches."""
    root = root.resolve()
    api_url = _check_api(api_url)
    _git_ready(root)
    if (STATUS_DIR / 'active.lock').exists():
        raise RuntimeError('Stage 7 đang xử lý tác vụ khác.')
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    for lock in STATUS_DIR.glob('auto_batch_*.lock'):
        raise RuntimeError(f'Có batch đang chạy hoặc còn lock: {lock.name}')
    batch = next_batch(root)
    if ((root / 'simulation_batches' / batch).exists()
            or (root / 'confirmed_batches' / batch).exists()
            or status_path(batch).exists()):
        raise RuntimeError('Batch đã tồn tại; không được ghi đè.')
    worker = root / 'pipeline' / 'stage7_batch5_worker.py'
    if not worker.is_file():
        raise RuntimeError('Không tìm thấy pipeline worker.')
    # Atomic global lock prevents two Streamlit sessions from launching together.
    try:
        fd = os.open(str(GLOBAL_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError('Có một phiên tự động đang chạy (hoặc lock còn sót).')
    child_lock = STATUS_DIR / f'auto_{batch}.lock'
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(batch)
        fd2 = os.open(str(child_lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd2, 'w') as stream:
            stream.write(batch)
        tmp = STATUS_DIR / f'.{batch}_{uuid.uuid4().hex}.tmp'
        tmp.write_text(json.dumps({'batch': batch, 'state': 'queued',
                                   'mode': 'TEST_ONLY_NO_PROMOTION',
                                   'created_by_auto_worker': True,
                                   'completed_frames': 0, 'completed_crops': 0},
                                  ensure_ascii=False), encoding='utf-8')
        tmp.replace(status_path(batch))
        cmd = [sys.executable, str(worker), '--batch', batch, '--api-url', api_url]
        kwargs = {'cwd': root, 'stdin': subprocess.DEVNULL, 'close_fds': True}
        if os.name == 'nt':
            kwargs['creationflags'] = (subprocess.CREATE_NEW_PROCESS_GROUP |
                                       subprocess.DETACHED_PROCESS)
        else:
            kwargs['start_new_session'] = True
        with (STATUS_DIR / f'auto_{batch}.log').open('ab') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        return batch, process.pid
    except Exception:
        child_lock.unlink(missing_ok=True)
        GLOBAL_LOCK.unlink(missing_ok=True)
        status_path(batch).unlink(missing_ok=True)
        raise


def _snapshot(root: Path, batch: str, completed: int) -> None:
    """Read ONLY an already-processed frame; worker remains sole writer."""
    if completed <= 0:
        return
    manifest_file = root / 'simulation_batches' / batch / 'manifest.json'
    if not manifest_file.is_file():
        return
    try:
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        if (manifest.get('batch_name') != batch or manifest.get('source_split') != 'data/train'
                or manifest.get('test_only') is not True):
            return
        frames = manifest.get('images', [])
        if not isinstance(frames, list) or not 0 < completed <= len(frames):
            return
        item = frames[completed - 1]
        from online_stream import _record_path, _render_image, _annotate
        sim = manifest_file.parent
        prediction = _record_path(sim, item, 'mobilenetv3')
        if not prediction.is_file():
            return
        record = json.loads(prediction.read_text(encoding='utf-8'))
        if str(record.get('image_id')) != str(item['image_id']):
            return
        image = _annotate(_render_image(sim, record), record['predictions'])
        st.image(image, caption=f"Ảnh đã xử lý: {completed}/{len(frames)}",
                 width='stretch')
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        st.caption('Chưa tải được ảnh gần nhất.')


def _safe_ui_status(value) -> str:
    """Mask internal batch sequence identifiers in non-technical status messages."""
    message = str(value)
    message = re.sub(r"(?i)\bbatch[_ -]?\d+\b", "dữ liệu", message)
    message = re.sub(r"(?i)\bđợt dữ liệu\s+\d+\b", "đợt dữ liệu", message)
    message = re.sub(r"(?i)\bstage\s*\d+\b", "quy trình huấn luyện", message)
    return message


def _job_overview(root: Path, batch: str, data: dict, *, include_frame: bool) -> None:
    state = str(data.get('state', 'not_started'))
    done = max(0, min(20, int(data.get('completed_frames') or 0)))
    a, b, c = st.columns(3)
    a.metric('Ảnh đã xử lý', f'{done}/20')
    b.metric('Ô đỗ đã xác minh', str(data.get('completed_crops', 0)))
    c.metric('Tiến độ', STATE_LABELS.get(state, state))
    st.progress(done / 20)
    if data.get('message'):
        st.caption(_safe_ui_status(data['message'])[:300])
    if state == 'failed':
        st.error(_safe_ui_status(data.get('error', 'Lỗi chưa rõ'))[:700])
    elif state == 'submitted':
        st.success('Đã gửi dữ liệu. Xem kết quả ở mục Huấn luyện lại.')
    elif state in ACTIVE_STATES:
        st.info('Đang xử lý tự động trên tiến trình nền.')
    if include_frame:
        _snapshot(root, batch, done)


def render_camera_pipeline(root: Path, api_url: str) -> None:
    st.markdown('#### Camera mô phỏng và thu thập dữ liệu')
    st.caption('Phát lại ảnh PKLot TRAIN (không phải camera thực). Worker thực hiện '
               '5 giây/frame → FastAPI → xác minh COCO Ground Truth → DVC → GitHub Actions. '
               'Không tự thay đổi Production.')

    @st.fragment(run_every='5s')
    def live_progress():
        batch, record = latest_status()
        st.markdown('**Lần thu thập gần nhất**')
        _job_overview(root, batch, record, include_frame=True)

    live_progress()
    next_name = next_batch(root)
    st.caption('Có thể bắt đầu thu thập dữ liệu mới khi cần. Kết quả huấn luyện trước vẫn được giữ nguyên.')
    with st.expander('Bắt đầu thu thập dữ liệu mới', expanded=False):
        expected = os.environ.get('MLOPS_CONTROL_PASSWORD', '')
        if len(expected) < 12:
            st.warning('Thiếu biến MLOPS_CONTROL_PASSWORD trong tiến trình Streamlit.')
            return
        pwd = st.text_input('Mật khẩu điều khiển', type='password', key='cam_auto_pwd')
        authorized = bool(pwd) and hmac.compare_digest(pwd.encode(), expected.encode())
        ack = st.checkbox('Tôi xác nhận đây là mô phỏng PKLot TRAIN và đồng ý '
                          'gửi dữ liệu mô phỏng lên GitHub/DVC để huấn luyện thử nghiệm khi đủ 20 ảnh hợp lệ.',
                          key='cam_auto_ack')
        reviewed = st.checkbox('Đã xem kết quả đánh giá của lần huấn luyện trước; '
                               'chấp nhận tạo một batch khác, không tự thay Production.',
                               key='cam_auto_reviewed')
        disabled = (not authorized or not ack or not reviewed or GLOBAL_LOCK.exists()
                    or (STATUS_DIR / 'active.lock').exists())
        if st.button('▶ Bắt đầu thu thập dữ liệu và huấn luyện lại',
                     type='primary', disabled=disabled, key='cam_auto_start_next'):
            try:
                batch, pid = start_next(root, api_url)
                st.success('Đã bắt đầu thu thập. Theo dõi tiến độ tại đây; không bấm lại.')
                st.rerun()
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                st.error(_safe_ui_status(exc)[:800])
        if not authorized:
            st.caption('Nhập đúng mật khẩu để bật nút chạy. Không nhập GitHub Token tại đây.')


def render_pipeline_summary(root: Path) -> None:
    """Read-only overview; starting capture belongs to the Camera tab."""
    st.markdown('##### Tiến độ thu thập dữ liệu')
    batch, record = latest_status()
    st.caption('Dữ liệu được thu thập từ mục Camera mô phỏng.')
    _job_overview(root, batch, record, include_frame=False)
    st.link_button('Mở GitHub Actions',
                   'https://github.com/thaibao3214/smart-parking-mlops/actions')
