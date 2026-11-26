"""Camera simulation UI embedded in dashboard/app.py.

Data provenance is preserved. Old PKLot test simulation and new PKLot train
simulation remain in separate directories. Neither is approved external data.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st
from stage7_review_actions import exclude_frame, read_exclusion_ids
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent

SOURCES = {
    'PKLot train — TEST_ONLY (mới)': {
        'stream': ROOT / 'simulation_train',
        'confirmed': ROOT / 'confirmed_train_sim',
        'origin': 'pklot_train_simulation',
        'source_split': 'data/train',
    },
    'PKLot test — phiên cũ (lưu bằng chứng)': {
        'stream': ROOT / 'data' / 'realtime_stream',
        'confirmed': ROOT / 'data' / 'confirmed_online',
        'origin': 'pklot_test_simulation',
        'source_split': 'data/test',
    },
}


def _frame_file(stream, item):
    name = item['file_name']
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('Tên file frame không hợp lệ')
    file_path = stream / 'images' / name
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return file_path


def _record_path(stream, item, model_type="mobilenetv3"):
    image_id = str(item['image_id'])
    if not image_id.isdigit():
        raise ValueError('image_id không hợp lệ')
    suffix = "" if model_type == "mobilenetv3" else f"_{model_type}"
    return stream / 'predictions' / f"{image_id}_{Path(item['file_name']).name}{suffix}.json"


def _frame_key(record):
    return f"{record['image_id']}_{Path(record['file_name']).stem}"


def _render_image(stream, record):
    # FastAPI returns bounding boxes in 640x640 coordinates.
    with Image.open(_frame_file(stream, record)) as image:
        return image.convert('RGB').resize((640, 640))


def _annotate(image, predictions):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for pred in predictions:
        x, y, w, h = [int(v) for v in pred['box']]
        color = '#16a34a' if pred['status'].lower() == 'empty' else '#dc2626'
        draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
        draw.text((x, max(y - 14, 0)), f"#{pred['id']} {pred['status']}", fill=color)
    return canvas


def _predict(stream, item, api_url, model_type, origin, source_split):
    src = _frame_file(stream, item)
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        files={'file': (src.name, src.read_bytes(), 'application/octet-stream')},
        data={'model_type': model_type},
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f'FastAPI HTTP {response.status_code}: {response.text[:400]}')
    result = response.json()
    predictions = result.get('predictions')
    if not isinstance(predictions, list):
        raise ValueError('API response thiếu predictions')
    return {
        'image_id': item['image_id'],
        'file_name': src.name,
        'source_file_name': item.get('source_file_name', src.name),
        'source_split': source_split,
        'origin': origin,
        'test_only': True,
        'model_type': model_type,
        'model_version': result.get('model_version'),
        'model_source': result.get('model_source'),
        'predictions': predictions,
        'empty_spaces': result.get('empty_spaces', 0),
        'occupied_spaces': result.get('occupied_spaces', 0),
        'occupancy_rate': result.get('occupancy_rate', 0),
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
    }


def _save_review(stream, confirmed, record, labels, source):
    """Save complete reviewed frame atomically; never overwrite past reviews."""
    target = confirmed / 'frames' / _frame_key(record)
    if target.exists():
        raise FileExistsError('Frame này đã được xác nhận; không ghi đè.')
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / ('.tmp_' + uuid.uuid4().hex)
    temp.mkdir()
    try:
        image = _render_image(stream, record)
        slots = []
        seen_ids = set()
        for pred in record['predictions']:
            slot_id = int(pred['id'])
            if slot_id in seen_ids:
                raise ValueError('Slot ID bị trùng')
            seen_ids.add(slot_id)
            label = labels.get(slot_id)
            if label not in ('Empty', 'Occupied'):
                raise ValueError(f'Chưa xác nhận slot {slot_id}')
            x, y, w, h = [int(v) for v in pred['box']]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(image.width, x + w), min(image.height, y + h)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f'BBox sai cho slot {slot_id}')
            crop_name = f'slot_{slot_id:03d}.png'
            image.crop((x0, y0, x1, y1)).resize((64, 64)).save(temp / crop_name)
            slots.append({
                'slot_id': slot_id, 'crop_file': crop_name,
                'bbox_xywh': [x0, y0, x1 - x0, y1 - y0],
                'predicted_label': pred['status'],
                'confirmed_label': label,
                'label_id': 0 if label == 'Empty' else 1,
            })
        payload = {
            'source_image': record['file_name'],
            'source_file_name': record.get('source_file_name', record['file_name']),
            'source_image_id': record['image_id'],
            'source_split': source['source_split'],
            'origin': source['origin'],
            'test_only': True,
            'production_approved': False,
            'model_type': record['model_type'],
            'model_version': record['model_version'],
            'model_source': record['model_source'],
            'review_method': 'manual_frame_review',
            'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
            'slots': slots,
        }
        (temp / 'labels.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        temp.rename(target)
    finally:
        if temp.exists():
            import shutil
            shutil.rmtree(temp)
    return target, len(slots)


def available_sources():
    """Expose immutable TEST_ONLY batches without touching the old simulations."""
    result = dict(SOURCES)
    for manifest_file in sorted((ROOT / 'simulation_batches').glob('*/manifest.json')):
        batch = manifest_file.parent.name
        if not batch.startswith('batch_'):
            continue
        result[f'PKLot train — {batch} (TEST_ONLY)'] = {
            'stream': manifest_file.parent,
            'confirmed': ROOT / 'confirmed_batches' / batch,
            'origin': 'pklot_train_simulation',
            'source_split': 'data/train',
        }
    return result


def render_online_stream(api_url='http://127.0.0.1:8000', model_type='mobilenetv3'):
    """Render one tab of the main Streamlit dashboard (no set_page_config here)."""
    st.subheader('📹 Camera mô phỏng — nhận diện mỗi 5 giây')
    st.caption('PKLot train/test là dữ liệu mô phỏng TEST_ONLY; không phải camera mới ngoài thực tế.')
    sources = available_sources()
    choice = st.selectbox('Nguồn luồng camera', list(sources), key='cam_source')
    source = sources[choice]
    stream = source['stream']
    confirmed = source['confirmed']
    manifest_path = stream / 'manifest.json'
    if not manifest_path.exists():
        st.info(
            'Chưa có phiên mô phỏng ở đây. Với PKLot train, chạy '
            '`python prepare_train_simulation.py --count 20` trước.'
        )
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        frames = manifest['images']
        if not isinstance(frames, list) or not frames:
            raise ValueError('manifest.images phải là danh sách khác rỗng')
        if choice.startswith('PKLot train') and (
            manifest.get('origin') != 'pklot_train_simulation'
            or manifest.get('source_split') != 'data/train'
            or manifest.get('test_only') is not True
        ):
            raise ValueError('Manifest PKLot train không đúng nguồn hoặc chưa đánh dấu TEST_ONLY')
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f'Manifest không hợp lệ: {exc}')
        return

    prefix = 'cam_' + ('test' if choice.startswith('PKLot test') else source['stream'].name)
    state_keys = ('running', 'index', 'success', 'failure')
    for name in state_keys:
        st.session_state.setdefault(prefix + '_' + name, False if name == 'running' else 0)
    st.caption(f'{len(frames)} frame · {source["source_split"]} · model: {model_type}')

    tab_camera, tab_review = st.tabs(['▶️ Phát camera', '✅ Xác nhận nhãn'])
    with tab_camera:
        c1, c2, c3 = st.columns(3)
        if c1.button('▶ Bắt đầu', key=prefix+'_start'):
            if st.session_state[prefix+'_index'] >= len(frames):
                st.session_state[prefix+'_index'] = 0
            st.session_state[prefix+'_running'] = True
        if c2.button('⏸ Tạm dừng', key=prefix+'_pause'):
            st.session_state[prefix+'_running'] = False
        if c3.button('🔄 Chạy lại luồng', key=prefix+'_reset'):
            st.session_state[prefix+'_running'] = False
            st.session_state[prefix+'_index'] = 0
            st.session_state[prefix+'_success'] = 0
            st.session_state[prefix+'_failure'] = 0

        @st.fragment(run_every='5s')
        def _camera_fragment():
            i = st.session_state[prefix+'_index']
            if not st.session_state[prefix+'_running']:
                st.info(f'Đang tạm dừng — {i}/{len(frames)} frame đã đi qua.')
                return
            if i >= len(frames):
                st.session_state[prefix+'_running'] = False
                st.success('Đã chạy hết phiên mô phỏng.')
                return
            item = frames[i]
            try:
                output = _record_path(stream, item, model_type)
                if output.exists():
                    record = json.loads(output.read_text(encoding='utf-8'))
                else:
                    record = _predict(stream, item, api_url, model_type,
                                      source['origin'], source['source_split'])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
                st.write(f'Frame {i+1}/{len(frames)}')
                left, right = st.columns([2, 1])
                left.image(_annotate(_render_image(stream, record), record['predictions']), width='stretch')
                right.metric('Tổng số ô', len(record['predictions']))
                right.metric('Ô trống', record.get('empty_spaces', 0))
                right.metric('Có xe', record.get('occupied_spaces', 0))
                right.caption(f"Model: {record.get('model_version')} ({record.get('model_source')})")
                st.session_state[prefix+'_success'] += 1
            except Exception as exc:
                st.error(f'Frame {i+1} lỗi: {exc}')
                st.session_state[prefix+'_failure'] += 1
            st.session_state[prefix+'_index'] = i + 1
            st.progress((i+1)/len(frames))
            st.caption('Thành công: '
                       f"{st.session_state[prefix+'_success']} · "
                       f"Lỗi: {st.session_state[prefix+'_failure']}")
            if i + 1 == len(frames):
                st.session_state[prefix+'_running'] = False
                st.success('Đã xử lý hết frame. Sang tab Xác nhận nhãn.')

        _camera_fragment()

    with tab_review:
        st.info('Phải tự xem và xác nhận từng nhãn. Không lấy nhãn COCO làm nhãn xác nhận.')
        if st.button('🔄 Làm mới danh sách', key=prefix+'_refresh'):
            st.rerun()
        available = [f for f in frames if _record_path(stream, f, model_type).is_file()]
        if not available:
            st.warning('Chưa có kết quả dự đoán trong phiên này.')
        else:
            idx = st.selectbox('Frame cần kiểm tra', range(len(available)),
                               format_func=lambda k: f"Frame {frames.index(available[k])+1:02d} — {available[k]['file_name']}",
                               key=prefix+'_review_frame')
            item = available[idx]
            record = json.loads(_record_path(stream, item, model_type).read_text(encoding='utf-8'))
            target = confirmed / 'frames' / _frame_key(record)
            try:
                excluded_ids = read_exclusion_ids(stream, confirmed, manifest) if stream.name.startswith('batch_') else set()
            except Exception as exc:
                st.error(f'Không đọc được exclusion record: {exc}')
                st.stop()
            is_excluded = str(record['image_id']) in excluded_ids
            if is_excluded:
                st.warning('Frame đã bị loại khỏi retraining. Ảnh gốc và dự đoán vẫn được giữ lại.')
            elif target.exists():
                prior = json.loads((target / 'labels.json').read_text(encoding='utf-8'))
                if prior.get('review_method') == 'pklot_coco_ground_truth':
                    st.success('Frame này đã xác minh tự động bằng nhãn PKLot COCO TRAIN (không phải duyệt thủ công).')
                else:
                    st.success('Frame này đã lưu nhãn xác nhận thủ công, không ghi đè.')
            else:
                if stream.name.startswith('batch_'):
                    st.markdown('**Quản lý chất lượng frame**')
                    reason = st.selectbox(
                        'Lý do loại frame',
                        ['Bounding box sai vị trí, không thể xác nhận tin cậy',
                         'Nhãn hoặc vị trí ô đỗ quá nhiều lỗi, cần loại frame',
                         'Ảnh không đủ chất lượng để xác nhận'],
                        key=f"{prefix}_reason_{record['image_id']}",
                    )
                    exclude_ack = st.checkbox(
                        'Tôi đã kiểm tra và xác nhận cần loại frame này khỏi retraining.',
                        key=f"{prefix}_exclude_ack_{record['image_id']}",
                    )
                    if st.button('🚫 Loại frame khỏi batch',
                                 key=f"{prefix}_exclude_{record['image_id']}",
                                 disabled=not exclude_ack):
                        try:
                            n = exclude_frame(stream, confirmed, manifest, item, reason)
                            st.success(f'Đã ghi nhận loại frame. Tổng frame bị loại: {n}.')
                            st.rerun()
                        except Exception as exc:
                            st.error(f'Không thể loại frame: {exc}')
                predictions = record.get('predictions', [])
                st.image(_annotate(_render_image(stream, record), predictions), width='stretch')
                rows = [
                    {'id': int(p['id']), 'predicted': p['status'], 'confirmed_label': p['status']}
                    for p in predictions
                ]
                edited = st.data_editor(
                    rows, key=f"{prefix}_edit_{record['image_id']}", hide_index=True,
                    num_rows='fixed', disabled=['id', 'predicted'],
                    column_config={
                        'id': st.column_config.NumberColumn('Slot ID'),
                        'predicted': st.column_config.TextColumn('AI dự đoán'),
                        'confirmed_label': st.column_config.SelectboxColumn(
                            'Nhãn xác nhận', options=['Empty', 'Occupied'], required=True
                        ),
                    }, use_container_width=True,
                )
                ack = st.checkbox('Tôi đã kiểm tra toàn bộ ô đỗ trong frame này.',
                                  key=f"{prefix}_ack_{record['image_id']}")
                if st.button('💾 Lưu nhãn xác nhận', key=prefix+'_save', disabled=not ack):
                    try:
                        records = edited.to_dict('records') if hasattr(edited, 'to_dict') else edited
                        labels = {int(r['id']): r['confirmed_label'] for r in records}
                        if len(labels) != len(predictions) or not predictions:
                            raise ValueError('Thiếu dự đoán hoặc thiếu nhãn xác nhận')
                        path, count = _save_review(stream, confirmed, record, labels, source)
                        st.success(f'Đã lưu {count} crop 64×64 vào {path}')
                    except Exception as exc:
                        st.error(str(exc))
        count = len(list((confirmed / 'frames').glob('*/labels.json')))
        st.metric('Số frame đã xác nhận trong nguồn này', count)
        if stream.name.startswith('batch_'):
            try:
                excluded = read_exclusion_ids(stream, confirmed, manifest)
                valid_reviewed = sum(
                    1 for item in frames
                    if str(item['image_id']) not in excluded and
                    (confirmed / 'frames' / f"{item['image_id']}_{Path(item['file_name']).stem}" / 'labels.json').is_file()
                )
                a, b, c = st.columns(3)
                a.metric('Frame hợp lệ đã duyệt', f'{valid_reviewed}/{len(frames) - len(excluded)}')
                b.metric('Frame loại trừ', len(excluded))
                c.metric('Frame chưa xử lý', len(frames) - valid_reviewed - len(excluded))
                if valid_reviewed + len(excluded) == len(frames):
                    st.success('Đã xử lý hết các frame. Có thể vào MLOps Control Center để Dry Run.')
            except Exception as exc:
                st.error(f'Lỗi kiểm tra tiến độ batch: {exc}')
        st.caption(f'Dữ liệu xác nhận: {confirmed.relative_to(ROOT)} · chỉ dùng cho mô phỏng TEST_ONLY.')
