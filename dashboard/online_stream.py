import json
import uuid
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
STREAM = ROOT / 'data' / 'realtime_stream'
PRED_DIR = STREAM / 'predictions'
CONFIRMED = ROOT / 'data' / 'confirmed_online'
MANIFEST = STREAM / 'manifest.json'
API_URL = 'http://127.0.0.1:8000'
MODEL_TYPE = 'mobilenetv3'

st.set_page_config(page_title='Smart Parking Online', page_icon='🚗', layout='wide')
st.title('🚗 Smart Parking – Online Simulation & Data Collection')
st.caption('Nhận diện mỗi 5 giây | Xác nhận nhãn thủ công trước khi đưa vào retraining')

if not MANIFEST.exists():
    st.error('Chưa có dữ liệu. Hãy chạy python prepare_online.py trước.')
    st.stop()

manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
frames = manifest['images']

for key, default in [('running', False), ('frame_index', 0),
                     ('success', 0), ('failure', 0)]:
    if key not in st.session_state:
        st.session_state[key] = default


def frame_path(item):
    name = Path(item['file_name']).name
    return STREAM / 'images' / name


def prediction_path(item):
    # UUID/image id avoids collisions if filenames differ only slightly.
    return PRED_DIR / f"{item['image_id']}_{Path(item['file_name']).name}.json"


def infer(item):
    path = frame_path(item)
    raw = path.read_bytes()
    response = requests.post(
        f'{API_URL}/predict',
        files={'file': (path.name, raw, 'image/jpeg')},
        data={'model_type': MODEL_TYPE},
        timeout=90,
    )
    response.raise_for_status()
    result = response.json()
    return {
        'image_id': item['image_id'],
        'file_name': path.name,
        'model_type': MODEL_TYPE,
        'model_version': result.get('model_version'),
        'model_source': result.get('model_source'),
        'predictions': result.get('predictions', []),
        'empty_spaces': result.get('empty_spaces'),
        'occupied_spaces': result.get('occupied_spaces'),
        'occupancy_rate': result.get('occupancy_rate'),
        'captured_at_utc': __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat(),
    }


def draw_boxes(image, predictions):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for p in predictions:
        x, y, w, h = map(int, p['box'])
        status = p['status']
        color = '#16A34A' if status == 'Empty' else '#DC2626'
        draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
        draw.text((x, max(0, y - 12)), f"#{p['id']}", fill=color)
    return canvas


def get_image(record):
    img = Image.open(STREAM / 'images' / Path(record['file_name']).name).convert('RGB')
    # Must match the existing FastAPI preprocessing exactly.
    return img.resize((640, 640))


def save_confirmed(record, labels):
    """Commit one reviewed frame atomically; never overwrite existing labels."""
    frame_key = f"{record['image_id']}_{Path(record['file_name']).stem}"
    dst = CONFIRMED / 'frames' / frame_key
    if dst.exists():
        raise FileExistsError('Frame này đã được xác nhận. Không ghi đè dữ liệu đã lưu.')

    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.parent / ('.tmp_' + uuid.uuid4().hex)
    temp.mkdir()
    try:
        image = get_image(record)
        confirmed_slots = []
        for prediction in record['predictions']:
            slot_id = int(prediction['id'])
            label = labels[slot_id]
            if label not in ('Empty', 'Occupied'):
                raise ValueError(f'Nhãn không hợp lệ cho slot {slot_id}')
            x, y, w, h = map(int, prediction['box'])
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(image.width, x + w), min(image.height, y + h)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f'Bounding box không hợp lệ: {slot_id}')
            crop = image.crop((x0, y0, x1, y1)).resize((64, 64))
            crop_name = f'slot_{slot_id:03d}.png'
            crop.save(temp / crop_name)
            confirmed_slots.append({
                'slot_id': slot_id,
                'crop_file': crop_name,
                'bbox_xywh': [x0, y0, x1 - x0, y1 - y0],
                'predicted_label': prediction['status'],
                'confirmed_label': label,
                'label_id': 0 if label == 'Empty' else 1,
            })

        payload = {
            'source_image': record['file_name'],
            'source_image_id': record['image_id'],
            'model_type': record['model_type'],
            'model_version': record['model_version'],
            'model_source': record['model_source'],
            'reviewed_at_utc': __import__('datetime').datetime.now(
                __import__('datetime').timezone.utc).isoformat(),
            'review_method': 'manual_frame_review',
            'slots': confirmed_slots,
        }
        (temp / 'labels.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        temp.rename(dst)
    finally:
        if temp.exists():
            import shutil
            shutil.rmtree(temp)
    return dst, len(confirmed_slots)


tab_stream, tab_review = st.tabs(['📹 Camera online', '✅ Xác nhận nhãn'])

with tab_stream:
    a, b, c = st.columns(3)
    with a:
        if st.button('▶ Bắt đầu'):
            if st.session_state.frame_index >= len(frames):
                st.session_state.frame_index = 0
            st.session_state.running = True
    with b:
        if st.button('⏸ Tạm dừng'):
            st.session_state.running = False
    with c:
        if st.button('🔄 Chạy lại'):
            st.session_state.running = False
            st.session_state.frame_index = 0
            st.session_state.success = 0
            st.session_state.failure = 0

    @st.fragment(run_every='5s')
    def camera():
        i = st.session_state.frame_index
        if not st.session_state.running:
            st.info(f'Đang tạm dừng — {i}/{len(frames)} frame đã xử lý.')
            return
        if i >= len(frames):
            st.session_state.running = False
            st.success('Đã xử lý hết luồng ảnh.')
            return

        item = frames[i]
        st.subheader(f'Frame {i + 1}/{len(frames)}')
        try:
            output = prediction_path(item)
            if output.exists():
                record = json.loads(output.read_text(encoding='utf-8'))
            else:
                record = infer(item)
                PRED_DIR.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                  encoding='utf-8')
            col_img, col_stats = st.columns([2, 1])
            with col_img:
                st.image(draw_boxes(get_image(record), record['predictions']),
                         width='stretch')
            with col_stats:
                st.metric('Tổng số ô', len(record['predictions']))
                st.metric('Ô trống', record['empty_spaces'])
                st.metric('Có xe', record['occupied_spaces'])
                st.caption(f"Model: {record['model_version']} ({record['model_source']})")
            st.session_state.success += 1
        except Exception as exc:
            st.error(f'Frame {i + 1} lỗi: {exc}')
            st.session_state.failure += 1
        st.session_state.frame_index += 1
        st.progress(st.session_state.frame_index / len(frames))
        st.caption(f"Thành công: {st.session_state.success} | Lỗi: {st.session_state.failure}")
        if st.session_state.frame_index == len(frames):
            st.session_state.running = False
            st.success('Đã xử lý 20 frame. Chuyển sang tab Xác nhận nhãn.')

    camera()

with tab_review:
    st.write('Chọn một frame đã dự đoán. Kiểm tra ảnh, sửa nhãn sai, rồi xác nhận lưu.')
    st.info('COCO ground truth không được sử dụng để điền nhãn xác nhận.')
    if st.button('🔄 Làm mới danh sách frame'):
        st.rerun()

    available = [item for item in frames if prediction_path(item).exists()]
    if not available:
        st.warning('Chưa có frame dự đoán. Hãy chạy camera trước.')
    else:
        selected_idx = st.selectbox(
            'Frame cần kiểm tra', range(len(available)),
            format_func=lambda i: f"Frame {frames.index(available[i]) + 1:02d} – {Path(available[i]['file_name']).name}"
        )
        item = available[selected_idx]
        record = json.loads(prediction_path(item).read_text(encoding='utf-8'))
        frame_key = f"{record['image_id']}_{Path(record['file_name']).stem}"
        confirmed_dir = CONFIRMED / 'frames' / frame_key
        if confirmed_dir.exists():
            st.success('Frame này đã được xác nhận và lưu. Không ghi đè nhãn cũ.')
        else:
            st.image(draw_boxes(get_image(record), record['predictions']), width='stretch')
            rows = [
                {'id': p['id'], 'predicted': p['status'],
                 'confirmed_label': p['status']} for p in record['predictions']
            ]
            edited = st.data_editor(
                rows, key=f"review_{record['image_id']}", hide_index=True,
                num_rows='fixed', disabled=['id', 'predicted'],
                column_config={
                    'id': st.column_config.NumberColumn('Slot ID'),
                    'predicted': st.column_config.TextColumn('AI dự đoán'),
                    'confirmed_label': st.column_config.SelectboxColumn(
                        'Nhãn xác nhận', options=['Empty', 'Occupied'], required=True
                    ),
                }, use_container_width=True,
            )
            acknowledged = st.checkbox(
                'Tôi đã kiểm tra ảnh và xác nhận nhãn của tất cả ô trong frame này.',
                key=f"ack_{record['image_id']}"
            )
            if st.button('💾 Lưu dữ liệu đã xác nhận', type='primary',
                         disabled=not acknowledged):
                try:
                    edited_rows = edited.to_dict('records') if hasattr(edited, 'to_dict') else edited
                    labels = {int(r['id']): r['confirmed_label'] for r in edited_rows}
                    if len(labels) != len(record['predictions']):
                        raise ValueError('Thiếu ô đỗ xe cần xác nhận')
                    location, n = save_confirmed(record, labels)
                    st.success(f'Đã lưu {n} ảnh crop 64×64 tại {location}')
                except Exception as exc:
                    st.error(str(exc))

    saved = list((CONFIRMED / 'frames').glob('*/labels.json'))
    st.metric('Số frame đã xác nhận', len(saved))
    st.caption('Dữ liệu xác nhận chỉ phục vụ retraining sau khi được kiểm tra và version bằng DVC.')
