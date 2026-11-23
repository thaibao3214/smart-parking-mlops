"""Password-gated, local-only Streamlit controller for Stage 7 TEST_ONLY batches."""
import hmac
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import streamlit as st

from stage7_control_worker import job_dir

_BATCH_PATTERN = re.compile(r'batch_[0-9]{3,6}\Z')


def discover_batches(root):
    base = root / 'simulation_batches'
    result = []
    for file in sorted(base.glob('batch_*/manifest.json')):
        name = file.parent.name
        if _BATCH_PATTERN.fullmatch(name):
            result.append(name)
    return result


def read_statuses():
    directory = job_dir()
    items = []
    if not directory.is_dir():
        return items
    for file in directory.glob('*.json'):
        try:
            record = json.loads(file.read_text(encoding='utf-8'))
            if isinstance(record, dict) and 'batch' in record:
                record['job_id'] = file.stem
                items.append(record)
        except (OSError, ValueError):
            pass
    return sorted(items, key=lambda item: item.get('updated_at', ''), reverse=True)[:10]


def marker_for(root, batch_id):
    return root / 'pipeline' / 'stage7_batches' / f'{batch_id}.json'


def _validate_batch_data(root, batch, data):
    """Check metadata and immutable marker without bypassing Stage 7 validation."""
    if (data.get('batch_name') != batch
            or data.get('mode') != 'TEST_ONLY_NO_PROMOTION'
            or data.get('production_approved') is not False):
        raise RuntimeError('Batch metadata fails TEST_ONLY policy.')
    batch_id = data.get('batch_id', '')
    if not re.fullmatch(r'[0-9a-f]{20}', str(batch_id)):
        raise RuntimeError('Invalid batch identifier.')
    marker = marker_for(root, batch_id)
    if marker.exists():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        # A marker for a different version of the data is NOT a successful check.
        if (saved.get('batch_id') != batch_id
                or saved.get('batch_name') != batch
                or saved.get('production_approved') is not False
                or saved.get('frames') != data.get('frames')
                or saved.get('crops') != data.get('crops')
                or sorted(saved.get('excluded_source_image_ids', []))
                   != sorted(data.get('excluded_source_image_ids', []))):
            raise RuntimeError('Existing marker and current batch do not match.')
        data['already_submitted'] = True
    else:
        data['already_submitted'] = False
    return data


def dry_run(root, batch):
    """Verify new or already-submitted batches; never launch retraining."""
    cmd = [sys.executable, '-u', str(root / 'pipeline' / 'stage7_auto_trigger.py'),
           '--once', '--batch', batch, '--dry-run']
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=180, check=False)
    output = (result.stdout + '\n' + result.stderr).strip()

    if result.returncode != 0:
        raise RuntimeError('Batch verification failed; see diagnostic below.\n' + output[-2500:])

    if ('STAGE7_BATCH_VERIFIED:' in result.stdout
            and 'STAGE7_DRY_RUN_NO_CHANGES' in result.stdout):
        line = next(x for x in result.stdout.splitlines()
                    if x.startswith('STAGE7_BATCH_VERIFIED:'))
        data = json.loads(line.partition(':')[2].strip())
        return _validate_batch_data(root, batch, data)

    if 'STAGE7_BATCH_ALREADY_SUBMITTED:' in result.stdout:
        # Auto-trigger intentionally skips printed verification for a submitted batch.
        # Re-run the SAME verify() logic in a read-only subprocess, not a fake PASS.
        check_code = (
            'import json, sys; from pathlib import Path; '
            'sys.path.insert(0, str(Path.cwd() / "pipeline")); '
            'from stage7_auto_trigger import verify; '
            'print("STAGE7_VERIFIED_JSON:" + json.dumps(verify(sys.argv[1])))'
        )
        check = subprocess.run([sys.executable, '-c', check_code, batch],
                               cwd=root, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=180, check=False)
        if check.returncode != 0:
            raise RuntimeError('Submitted batch failed fresh verification.\n'
                               + (check.stdout + '\n' + check.stderr)[-2500:])
        lines = [line for line in check.stdout.splitlines()
                 if line.startswith('STAGE7_VERIFIED_JSON:')]
        if len(lines) != 1:
            raise RuntimeError('Unable to read submitted batch verification result.')
        data = json.loads(lines[0].partition(':')[2])
        data = _validate_batch_data(root, batch, data)
        if not data['already_submitted']:
            raise RuntimeError('Batch reported submitted but matching marker is absent.')
        return data

    raise RuntimeError('Unexpected Stage 7 dry-run response.\n' + output[-2500:])


def _is_authorized():
    expected = os.environ.get('MLOPS_CONTROL_PASSWORD', '')
    if len(expected) < 12:
        return False
    supplied = st.session_state.get('stage7_password', '')
    return hmac.compare_digest(expected.encode('utf-8'), supplied.encode('utf-8'))


def render_stage7_control(root: Path):
    st.divider()
    st.markdown('#### Continuous Training — Điều khiển Stage 7')
    st.caption('Chỉ điều khiển TEST_ONLY. Nút này chạy pipeline thật trên máy đang chạy Streamlit; '
               'sau khi push marker, GitHub Actions mới thực hiện retraining trên Kaggle GPU. '
               'Không tự chuyển model sang Production.')
    if not (root / 'pipeline' / 'stage7_auto_trigger.py').is_file():
        st.warning('Không tìm thấy pipeline/stage7_auto_trigger.py trong project.')
        return
    if len(os.environ.get('MLOPS_CONTROL_PASSWORD', '')) < 12:
        st.warning('Chức năng điều khiển đang khóa. Cấu hình MLOPS_CONTROL_PASSWORD (ít nhất 12 ký tự) '
                   'trong môi trường của tiến trình Streamlit rồi khởi động lại ứng dụng.')
        return
    st.text_input('Mật khẩu điều khiển Stage 7', type='password', key='stage7_password',
                  help='Không phải DagsHub/GitHub token. Không ghi vào mã nguồn.')
    if not _is_authorized():
        st.info('Nhập mật khẩu điều khiển để kiểm tra batch và kích hoạt retraining.')
        return

    batches = discover_batches(root)
    if not batches:
        st.info('Chưa có batch trong simulation_batches/. Tạo batch mới ở máy rồi xác nhận nhãn ở tab Camera.')
        return
    batch = st.selectbox('Batch mô phỏng PKLot train', batches, index=len(batches)-1, key='stage7_selected_batch')
    st.caption('Mỗi batch mới được gửi MỘT LẦN từ Streamlit. GitHub Actions tự khởi chạy bởi Git push marker; '
               'KHÔNG cần nhấn Run workflow thủ công. Batch đã gửi không gửi lại từ đây.')

    previous = st.session_state.get('stage7_checked')
    if previous and previous.get('batch_name') != batch:
        st.session_state.pop('stage7_checked', None)
        previous = None

    if st.button('🔍 Kiểm tra dữ liệu (Dry Run)', key='stage7_verify'):
        st.session_state.pop('stage7_checked', None)
        try:
            with st.spinner('Đang kiểm tra nguồn dữ liệu, số crop và exclusions...'):
                data = dry_run(root, batch)
            st.session_state['stage7_checked'] = data
            previous = data
            if data.get('already_submitted'):
                st.info('ALREADY SUBMITTED — Batch đã gửi; kiểm tra dữ liệu PASS. Không chạy retraining trùng.')
            else:
                st.success('Batch hợp lệ — Dry Run PASS')
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            st.error(str(exc)[:2700])
            previous = None

    if previous:
        batch_id = previous['batch_id']
        col1, col2, col3 = st.columns(3)
        col1.metric('Frames hợp lệ', previous.get('frames', '—'))
        col2.metric('Crops', previous.get('crops', '—'))
        col3.metric('Loại trừ', len(previous.get('excluded_source_image_ids', [])))
        st.caption(f'Batch ID: {batch_id} · Mode: {previous.get("mode")}')
        marker = marker_for(root, batch_id)
        if previous.get('already_submitted') or marker.exists():
            st.info('Batch đã được gửi (immutable marker tồn tại). Không thể trigger trùng.')
        elif (job_dir() / 'active.lock').exists():
            st.warning('Đang có một phiên kích hoạt Stage 7 khác. Xem trạng thái bên dưới.')
        else:
            acknowledge = st.checkbox('Tôi đã kiểm tra nhãn/bounding box và đồng ý gửi batch TEST_ONLY lên DVC/GitHub.',
                                      key='stage7_ack')
            if st.button('🚀 Bắt đầu retraining', type='primary', disabled=not acknowledge, key='stage7_launch'):
                # Recheck immutable batch and metadata directly before submitting.
                try:
                    fresh = dry_run(root, batch)
                    if fresh['batch_id'] != batch_id or marker_for(root, batch_id).exists():
                        raise RuntimeError('Batch đã thay đổi hoặc đã gửi; kiểm tra lại.')
                    if (job_dir() / 'active.lock').exists():
                        raise RuntimeError('Đã có một phiên Stage 7 đang chạy.')
                    # Fail before starting a detached worker when GitHub code is stale.
                    codecheck = subprocess.run(
                        [sys.executable, '-c',
                         'import sys; sys.path.insert(0, "pipeline"); '
                         'from stage7_auto_trigger import verify_code_ready; verify_code_ready()'],
                        cwd=root, capture_output=True, text=True,
                        encoding='utf-8', errors='replace', timeout=20, check=False)
                    if codecheck.returncode != 0:
                        raise RuntimeError('Chưa đồng bộ code Stage 7 lên GitHub:\n' +
                                           (codecheck.stderr or codecheck.stdout)[-1300:])
                    directory = job_dir()
                    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                    job_id = uuid.uuid4().hex[:24]
                    worker = Path(__file__).resolve().parent / 'stage7_control_worker.py'
                    kwargs = {'cwd': root, 'stdin': subprocess.DEVNULL,
                              'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL,
                              'close_fds': True}
                    if os.name == 'nt':
                        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                    else:
                        kwargs['start_new_session'] = True
                    subprocess.Popen([sys.executable, str(worker), '--batch', batch,
                                      '--job-id', job_id], **kwargs)
                    st.session_state.pop('stage7_checked', None)
                    st.success(f'Đã gửi yêu cầu chạy cục bộ: {job_id}. '
                               'Nhấn “Cập nhật trạng thái” để xem kết quả; '
                               'GitHub Actions chỉ bắt đầu sau khi marker được push.')
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    st.error(f'Không thể kích hoạt: {exc}')

    st.markdown('##### Trạng thái các lần kích hoạt từ Streamlit')
    if st.button('🔄 Cập nhật trạng thái', key='stage7_reload_jobs'):
        st.rerun()
    statuses = read_statuses()
    if not statuses:
        st.caption('Chưa có lần kích hoạt nào từ giao diện này.')
    for record in statuses[:5]:
        st.write(f"**{record.get('batch')}** · {record.get('state')} · {record.get('updated_at', '—')}")
        if record.get('error'):
            st.warning(str(record['error'])[:400])
        log_path = job_dir() / f"{record['job_id']}.log"
        if log_path.is_file():
            with st.expander(f"Nhật ký {record['job_id']}"):
                content = log_path.read_text(encoding='utf-8', errors='replace')
                st.code(content[-7000:] or 'Đang khởi động...', language='text')
    st.caption('“submitted” chỉ xác nhận đã push marker, không phải Kaggle/MLflow đã hoàn tất. '
               'Theo dõi các bước tiếp theo ở bảng GitHub Actions và MLflow phía trên.')
