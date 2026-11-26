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
from stage7_monitor import matching_run, summary_from_artifact

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



def launch_worker(root: Path, batch: str, mode: str, batch_id: str = ''):
    """Start one local background worker. The worker performs final validation."""
    directory = job_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (directory / 'active.lock').exists():
        raise RuntimeError('Một phiên Stage 7 khác đang chạy.')
    job_id = uuid.uuid4().hex[:24]
    worker = Path(__file__).resolve().parent / 'stage7_control_worker.py'
    kwargs = {'cwd': root, 'stdin': subprocess.DEVNULL,
              'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL,
              'close_fds': True}
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs['start_new_session'] = True
    args = [sys.executable, str(worker), '--batch', batch, '--job-id', job_id,
            '--mode', mode]
    if mode == 'dispatch':
        args.extend(['--batch-id', batch_id])
    subprocess.Popen(args, **kwargs)
    return job_id


def render_stage7_control(root: Path):
    st.divider()
    st.markdown('#### Continuous Training — Điều khiển Stage 7')
    st.caption('Chỉ điều khiển TEST_ONLY. Nút này chạy pipeline thật trên máy đang chạy Streamlit; '
               'sau khi push marker, GitHub Actions mới thực hiện retraining trên Kaggle GPU. '
               'Không tự chuyển model sang Production.')
    if not (root / 'pipeline' / 'stage7_auto_trigger.py').is_file():
        st.warning('Không tìm thấy pipeline/stage7_auto_trigger.py trong project.')
        return
    from stage7_unified_camera_ui import render_pipeline_summary
    render_pipeline_summary(root)
    if len(os.environ.get('MLOPS_CONTROL_PASSWORD', '')) < 12:
        st.warning('Chức năng điều khiển đang khóa. Cấu hình MLOPS_CONTROL_PASSWORD (ít nhất 12 ký tự) '
                   'trong môi trường của tiến trình Streamlit rồi khởi động lại ứng dụng.')
        return
    st.text_input('Mật khẩu điều khiển Stage 7', type='password', key='stage7_password',
                  help='Không phải DagsHub/GitHub token. Không ghi vào mã nguồn.')
    if not _is_authorized():
        st.info('Nhập mật khẩu điều khiển để kiểm tra batch và kích hoạt retraining.')
        return

    with st.expander('Công cụ điều khiển thủ công (nâng cao)', expanded=False):
        batches = discover_batches(root)
        if not batches:
            st.info('Chưa có batch trong simulation_batches/. Tạo batch mới ở máy rồi xác nhận nhãn ở tab Camera.')
            return
        batch = st.selectbox('Batch mô phỏng PKLot train', batches, index=len(batches)-1, key='stage7_selected_batch')
        st.caption('Batch mới: push marker để GitHub tự chạy. Batch đã gửi: có thể chủ động chạy lại TEST ONLY '
                   'bằng GitHub API từ Streamlit (cần GH_TOKEN). Không thay đổi Production.')

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
                    st.info('ALREADY SUBMITTED — Dữ liệu đã gửi và kiểm tra PASS. Có thể chọn chạy lại TEST ONLY có chủ đích bên dưới.')
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
                st.info('Batch đã gửi: không push DVC/Git lại. Có thể dispatch workflow TEST ONLY hiện có.')
                if not os.environ.get('GH_TOKEN', '').strip():
                    st.warning('Thiếu GH_TOKEN trong môi trường Streamlit. Cần token GitHub có quyền Actions: write '
                               'và Contents: read cho repository để điều khiển workflow từ đây.')
                acknowledge_rerun = st.checkbox(
                    'Tôi đồng ý chạy LẠI TEST ONLY trên Kaggle GPU cho batch đã gửi; '
                    'không thay đổi Production.', key='stage7_dispatch_ack')
                if st.button('🚀 Chạy retraining trên GitHub từ Streamlit', type='primary',
                             disabled=not acknowledge_rerun or not os.environ.get('GH_TOKEN', '').strip(),
                             key='stage7_dispatch'):
                    try:
                        fresh = dry_run(root, batch)
                        if not fresh.get('already_submitted') or fresh['batch_id'] != batch_id:
                            raise RuntimeError('Batch/marker không còn khớp. Hãy Dry Run lại.')
                        job_id = launch_worker(root, batch, 'dispatch', batch_id)
                        st.success(f'Đã tạo yêu cầu điều khiển {job_id}. Nhấn “Cập nhật trạng thái” '
                                   'để xem GitHub đã chấp nhận workflow_dispatch hay chưa.')
                    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                        st.error(f'Không thể điều khiển workflow: {exc}')
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
                        job_id = launch_worker(root, batch, 'push')
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
        if record.get('workflow_url'):
            st.link_button('Mở workflow GitHub', record['workflow_url'])
        if record.get('error'):
            st.warning(str(record['error'])[:400])
        log_path = job_dir() / f"{record['job_id']}.log"
        if log_path.is_file():
            with st.expander(f"Nhật ký {record['job_id']}"):
                content = log_path.read_text(encoding='utf-8', errors='replace')
                st.code(content[-7000:] or 'Đang khởi động...', language='text')
    st.markdown('##### Giám sát Retraining trên GitHub Actions')
    dispatched = [r for r in statuses if r.get('state') == 'dispatched' and r.get('batch_id')]
    if not dispatched:
        st.caption('Chưa có yêu cầu workflow_dispatch thành công để đối chiếu.')
    elif not os.environ.get('GH_TOKEN', '').strip():
        st.warning('Để kiểm tra run trên GitHub, cần GH_TOKEN có Actions: read.')
    else:
        labels = [f"{r['batch']} · {r.get('updated_at', '—')} · {r['job_id'][:8]}" for r in dispatched]
        chosen_index = st.selectbox('Lần kích hoạt cần giám sát', range(len(dispatched)),
                                    format_func=lambda i: labels[i], key='stage7_monitored_job')
        chosen = dispatched[chosen_index]
        ident = chosen['job_id']
        if st.button('🔎 Kiểm tra run trên GitHub', key='stage7_lookup_run'):
            try:
                with st.spinner('Đối chiếu Batch ID với nhật ký xác minh trên GitHub...'):
                    run = matching_run(chosen['batch'], chosen['batch_id'],
                                       chosen['updated_at'], os.environ['GH_TOKEN'].strip())
                st.session_state['stage7_confirmed_run'] = {'job_id': ident, 'run': run}
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f'Không kiểm tra được: {exc}')
        current = st.session_state.get('stage7_confirmed_run', {})
        if current.get('job_id') == ident:
            run = current.get('run')
            if not run:
                st.warning('Chưa xác minh được run tương ứng. Có thể job đang chạy/chưa có log. '
                           'Không tự gán một run bất kỳ cho batch.')
            else:
                st.success(f"Đã xác minh Batch ID trong job log · Run #{run['id']}")
                c1, c2, c3 = st.columns(3)
                c1.metric('Run ID', run['id'])
                c2.metric('GitHub status', run.get('status') or '—')
                c3.metric('Conclusion', run.get('conclusion') or '—')
                if run.get('url'):
                    st.link_button('Mở đúng GitHub run', run['url'])
                st.info('Workflow success là thành công kỹ thuật; các model vẫn có thể Gate FAIL. '
                        'TEST ONLY không tự promotion sang Production.')
                if run.get('status') == 'completed':
                    st.caption('Chỉ tải artifact REPORT nhỏ (không tải checkpoint). Run cũ chưa có REPORT: xem GitHub Actions → Summary.')
                    if st.button('📊 Lấy kết quả Evaluation Gate', key='stage7_fetch_metrics'):
                        try:
                            with st.spinner('Đọc artifact REPORT nhỏ từ GitHub...'):
                                summary, report = summary_from_artifact(run['id'], os.environ['GH_TOKEN'].strip())
                            st.session_state['stage7_run_report'] = {'run_id': run['id'],
                                                                      'summary': summary, 'report': report}
                        except (OSError, ValueError, RuntimeError) as exc:
                            st.error(f'Không đọc được Evaluation Gate: {exc}')
                    metrics = st.session_state.get('stage7_run_report', {})
                    if metrics.get('run_id') == run['id']:
                        st.markdown('###### Kết quả Evaluation Gate (đọc từ artifact của run đã xác minh)')
                        if metrics.get('summary'):
                            st.markdown(metrics['summary'])
                            st.download_button(
                                '📥 Tải báo cáo Evaluation Gate (.md)',
                                data=metrics['summary'].encode('utf-8'),
                                file_name=f"stage7_run_{run['id']}_evaluation.md",
                                mime='text/markdown',
                                key='stage7_download_summary')
                        if metrics.get('report') is not None:
                            with st.expander('Xem báo cáo JSON gốc'):
                                st.json(metrics['report'])
                        st.caption('Gate FAIL nghĩa là candidate chưa đạt điều kiện; '
                                   'workflow SUCCESS nghĩa là pipeline chạy thành công. '
                                   'Không promotion Production trong TEST ONLY.')
    st.caption('“submitted” xác nhận push marker; “dispatched” xác nhận GitHub nhận yêu cầu, không phải Kaggle/MLflow đã hoàn tất. '
               'Theo dõi các bước tiếp theo ở bảng GitHub Actions và MLflow phía trên.')
