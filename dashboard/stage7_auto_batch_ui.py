"""Password-protected Streamlit UI for a single auto TEST_ONLY batch_005 run."""
import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

STATUS_DIR = Path.home() / '.smartparking_stage7_control'
STATUS_FILE = STATUS_DIR / 'auto_batch_005.json'
LOCK_FILE = STATUS_DIR / 'auto_batch_005.lock'


def read_status():
    if not STATUS_FILE.exists():
        return {}
    try:
        return json.loads(STATUS_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {'state': 'unknown', 'message': 'Cannot load progress file'}


def start(root, api_url):
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic lock protects rapid double-click and two Streamlit sessions.
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError('Batch 005 is running (or lock remained after an interruption)')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write('running\n')
        cmd = [sys.executable, str(root / 'pipeline' / 'stage7_batch5_worker.py'),
               '--api-url', api_url]
        kwargs = {'cwd': root, 'stdin': subprocess.DEVNULL, 'stdout': subprocess.DEVNULL,
                  'stderr': subprocess.DEVNULL, 'close_fds': True}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs['start_new_session'] = True
        process = subprocess.Popen(cmd, **kwargs)
        return process.pid
    except Exception:
        LOCK_FILE.unlink(missing_ok=True)
        raise


def render_auto_batch_005(root):
    st.markdown('##### 🤖 Batch 005 — mô phỏng và retraining tự động')
    st.caption('Tự chọn 20 ảnh PKLot TRAIN mới → FastAPI (5 giây/frame) → xác minh COCO Ground Truth '
               '→ Stage 7 Dry Run → DVC/GitHub Actions. Không dùng dự đoán AI làm nhãn đúng. '
               'Không thay đổi Production.')
    current = read_status()
    state = current.get('state', 'not_started')
    done = int(current.get('completed_frames', 0))
    cols = st.columns(3)
    cols[0].metric('Frame', f'{done}/20')
    cols[1].metric('Crop GT', current.get('completed_crops', 0))
    cols[2].metric('Trạng thái', state)
    st.progress(max(0.0, min(1.0, done/20)))
    if current.get('message'):
        st.caption(str(current['message'])[:500])
    if current.get('batch_id'):
        st.code(str(current['batch_id']), language='text')
    if state == 'submitted':
        st.success('Đã push DVC/Git: GitHub Actions sẽ chạy Kaggle/MLflow/Gate. '
                   'Đây KHÔNG phải kết quả Evaluation Gate.')
    elif state == 'failed':
        st.error('Đã dừng an toàn: ' + str(current.get('error', 'Unknown'))[:700])
        st.info('Dữ liệu tạo dở được giữ lại để kiểm tra/tiếp tục; không tự xóa hay ghi đè.')
    elif LOCK_FILE.exists():
        st.info('Đang xử lý trên tiến trình nền; có thể đóng trang Streamlit và xem lại sau.')
    if st.button('🔄 Làm mới tiến độ Batch 005', key='batch5_refresh'):
        st.rerun()
    with st.expander('Bắt đầu một lần — Batch 005 TEST ONLY', expanded=not current):
        api = st.text_input('Địa chỉ FastAPI trên máy này', 'http://127.0.0.1:8000', key='batch5_api')
        approved = st.checkbox('Tôi đồng ý dùng COCO TRAIN Ground Truth làm nhãn mô phỏng '
                               'và cho phép gửi batch TEST ONLY lên DVC/GitHub khi vượt kiểm tra.',
                               key='batch5_ack')
        disabled = LOCK_FILE.exists() or state == 'submitted' or not approved
        if st.button('▶ Tạo và chạy tự động Batch 005', type='primary',
                     disabled=disabled, key='batch5_start'):
            try:
                from urllib.parse import urlparse
                parsed = urlparse(api)
                if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
                        or parsed.path not in ('', '/') or parsed.username or parsed.password):
                    raise ValueError('FastAPI phải là HTTP localhost / 127.0.0.1')
                if current.get('state') == 'submitted':
                    raise RuntimeError('Batch 005 đã được gửi')
                pid = start(root, api.strip())
                st.success(f'Đã khởi động Batch 005, tiến trình {pid}. Nhấn Làm mới tiến độ để theo dõi.')
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
    st.divider()
