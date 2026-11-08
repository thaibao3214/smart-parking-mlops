
import os
from pathlib import Path
from io import BytesIO

import streamlit as st
import requests
import pandas as pd

from PIL import Image, ImageDraw, ImageFont


# =====================================================
# 1. CAU HINH
# =====================================================

st.set_page_config(
    page_title="Smart Parking MLOps",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_API_URL = "http://127.0.0.1:8000"

MODEL_OPTIONS = {
    "MobileNetV3-Small": "mobilenetv3",
    "ResNet18": "resnet18",
    "Optimized Custom CNN": "custom_cnn"
}

CHART_PATHS = {
    "MobileNetV3-Small":
        BASE_DIR / "mobilenet_model" / "mobilenet_v3_metrics.png",

    "ResNet18":
        BASE_DIR / "resnet_model" / "resnet18_metrics.png",

    "Optimized Custom CNN":
        BASE_DIR / "cnn_model" / "metrics.png"
}


# =====================================================
# 2. CSS
# =====================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        padding: 24px;
        border-radius: 14px;
        background: linear-gradient(120deg, #123c69, #227c70);
        color: white;
        margin-bottom: 18px;
    }

    .main-title h1 {
        color: white;
        font-size: 30px;
        margin-bottom: 8px;
    }

    .main-title p {
        color: #e8f2f5;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================

def normalize_api_url(url):
    url = url.strip().rstrip("/")

    if url.endswith("/predict"):
        url = url[:-8]

    return url


def get_api_health(api_base_url):
    try:
        response = requests.get(
            f"{api_base_url}/health",
            timeout=8
        )
        response.raise_for_status()
        return response.json(), None

    except requests.RequestException as e:
        return None, str(e)


def call_predict_api(
    api_base_url,
    filename,
    image_bytes,
    model_type
):
    files = {
        "file": (
            filename,
            image_bytes,
            "application/octet-stream"
        )
    }

    data = {
        "model_type": model_type
    }

    response = requests.post(
        f"{api_base_url}/predict",
        files=files,
        data=data,
        timeout=120
    )

    if not response.ok:
        try:
            detail = response.json().get(
                "detail",
                response.text
            )
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"API HTTP {response.status_code}: {detail}"
        )

    result = response.json()

    if "error" in result:
        raise RuntimeError(result["error"])

    return result


def draw_predictions(image, predictions):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    for p in predictions:
        x, y, w, h = p["box"]

        status = p["status"]

        if status.lower() == "empty":
            color = "#16A34A"
            label = "EMPTY"
        else:
            color = "#DC2626"
            label = "OCCUPIED"

        draw.rectangle(
            [x, y, x + w, y + h],
            outline=color,
            width=3
        )

        # Ve nhan tren tung o do
        text_y = max(0, y - 17)

        draw.rectangle(
            [x, text_y, x + 78, text_y + 17],
            fill=color
        )

        draw.text(
            (x + 3, text_y + 2),
            label,
            fill="white"
        )

    return draw_img


def show_prediction_metrics(result):
    total = result.get("total_spaces", 0)
    empty = result.get("empty_spaces", 0)
    occupied = result.get("occupied_spaces", 0)
    rate = result.get("occupancy_rate", 0)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Tổng ô đỗ", total)
    c2.metric("Ô trống", empty)
    c3.metric("Có xe", occupied)
    c4.metric("Tỷ lệ lấp đầy", f"{rate:.2f}%")

    st.progress(
        max(0, min(100, int(round(rate))))
    )

    return empty, occupied


def show_occupancy_chart(empty, occupied):
    df = pd.DataFrame({
        "Trạng thái": ["Empty", "Occupied"],
        "Số lượng": [empty, occupied]
    })

    if empty + occupied == 0:
        st.info("Không có ô đỗ hợp lệ để vẽ biểu đồ.")
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.pie(
        df["Số lượng"],
        labels=df["Trạng thái"],
        colors=["#16A34A", "#DC2626"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Phân bố trạng thái ô đỗ")

    st.pyplot(fig)

    plt.close(fig)


# =====================================================
# 4. SESSION STATE
# =====================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_image" not in st.session_state:
    st.session_state.prediction_image = None

if "prediction_key" not in st.session_state:
    st.session_state.prediction_key = None


# =====================================================
# 5. HEADER
# =====================================================

st.markdown("""
<div class="main-title">
    <h1>🚗 SMART PARKING MLOPS</h1>
    <p>
        Hệ thống giám sát bãi đỗ xe thông minh
        sử dụng Deep Learning, FastAPI, MLflow và DagsHub
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 6. SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Cấu hình hệ thống")

api_url_input = st.sidebar.text_input(
    "FastAPI Endpoint",
    DEFAULT_API_URL
)

api_base_url = normalize_api_url(api_url_input)

selected_model_name = st.sidebar.selectbox(
    "Chọn mô hình AI",
    list(MODEL_OPTIONS.keys())
)

model_type = MODEL_OPTIONS[selected_model_name]

st.sidebar.divider()

st.sidebar.subheader("🌐 Trạng thái FastAPI")

health_data, health_error = get_api_health(api_base_url)

if health_data:
    st.sidebar.success("FastAPI đang hoạt động")

    st.sidebar.write(
        "**Model source:**",
        health_data.get("model_source", "Unknown")
    )

    st.sidebar.write(
        "**Version:**",
        health_data.get("model_version", "Unknown")
    )

    st.sidebar.write(
        "**Device:**",
        health_data.get("device", "Unknown")
    )

    st.sidebar.write(
        "**Registry alias:**",
        health_data.get("registry_alias", "Unknown")
    )

    if health_data.get("model_source") == "DagsHub Registry":
        st.sidebar.success(
            "Đang sử dụng Model Registry Production"
        )
    else:
        st.sidebar.warning(
            "Đang sử dụng Local fallback"
        )

else:
    st.sidebar.error("Không kết nối được FastAPI")
    st.sidebar.caption(health_error)

st.sidebar.divider()

st.sidebar.subheader("📦 Thông tin dự án")

st.sidebar.markdown("""
**Dataset:** PKLot

**Framework:** PyTorch

**Model Registry:** MLflow + DagsHub

**Data Versioning:** DVC

**API:** FastAPI

**Dashboard:** Streamlit

**Deployment:** Local CPU
""")


# =====================================================
# 7. TABS
# =====================================================

tab1, tab2 = st.tabs([
    "📊 Giám sát bãi đỗ xe",
    "📈 Báo cáo và so sánh mô hình"
])


# =====================================================
# 8. TAB 1 - PREDICTION
# =====================================================

with tab1:

    st.subheader("📷 Hình ảnh bãi đỗ xe")

    uploaded_file = st.file_uploader(
        "Tải ảnh thuộc tập test PKLot",
        type=["jpg", "jpeg", "png"],
        help=(
            "Tên file ảnh phải tồn tại trong "
            "data/test/_annotations.coco.json"
        )
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

        source_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        source_image = source_image.resize((640, 640))

        # Dinh danh theo anh va model
        import hashlib

        image_hash = hashlib.md5(
            image_bytes
        ).hexdigest()

        current_key = (
            image_hash,
            model_type,
            api_base_url
        )

        if st.session_state.prediction_key != current_key:
            st.session_state.prediction_result = None
            st.session_state.prediction_image = None

        col_image, col_info = st.columns([2, 1])

        with col_image:
            st.image(
                source_image,
                caption=f"Ảnh gốc: {filename}",
                width="stretch"
            )

        with col_info:
            st.markdown("### Thông tin ảnh")

            st.write("**Tên file:**", filename)
            st.write("**Mô hình:**", selected_model_name)
            st.write(
                "**Kích thước xử lý:** 640 × 640"
            )

            if health_data:
                st.write(
                    "**Registry Version:**",
                    health_data.get(
                        "model_version", "Unknown"
                    )
                )

        if st.button(
            "🚀 Chạy suy luận AI",
            type="primary",
            disabled=health_data is None
        ):

            with st.spinner(
                "Đang phân tích trạng thái bãi đỗ xe..."
            ):

                try:
                    result = call_predict_api(
                        api_base_url,
                        filename,
                        image_bytes,
                        model_type
                    )

                    annotated_image = draw_predictions(
                        source_image,
                        result.get("predictions", [])
                    )

                    st.session_state.prediction_result = result
                    st.session_state.prediction_image = annotated_image
                    st.session_state.prediction_key = current_key

                    st.success(
                        "Nhận diện bãi đỗ xe thành công!"
                    )

                except Exception as e:
                    st.error(str(e))

        result = st.session_state.prediction_result

        if (
            result is not None
            and st.session_state.prediction_key == current_key
        ):

            st.divider()

            st.subheader("📊 Kết quả nhận diện")

            empty, occupied = show_prediction_metrics(
                result
            )

            result_col1, result_col2 = st.columns(
                [2, 1]
            )

            with result_col1:

                st.image(
                    st.session_state.prediction_image,
                    caption=(
                        "Kết quả nhận diện: "
                        "Xanh = Empty | Đỏ = Occupied"
                    ),
                    width="stretch"
                )

            with result_col2:

                st.markdown(
                    "### Tỷ lệ trạng thái ô đỗ"
                )

                show_occupancy_chart(
                    empty,
                    occupied
                )

            st.divider()

            st.subheader("🔍 Thông tin model thực hiện dự đoán")

            mc1, mc2, mc3 = st.columns(3)

            mc1.metric(
                "Model",
                result.get("model_used", "Unknown")
            )

            mc2.metric(
                "Version",
                str(result.get("model_version", "Unknown"))
            )

            mc3.metric(
                "Nguồn model",
                result.get("model_source", "Unknown")
            )

            with st.expander(
                "📋 Xem chi tiết từng ô đỗ"
            ):

                predictions = result.get(
                    "predictions", []
                )

                if predictions:
                    df_slots = pd.DataFrame(
                        predictions
                    )

                    st.dataframe(
                        df_slots,
                        width="stretch",
                        hide_index=True
                    )

    else:
        st.info(
            "Vui lòng tải lên một ảnh từ data/test "
            "để bắt đầu nhận diện."
        )


# =====================================================
# 9. TAB 2 - MODEL COMPARISON
# =====================================================

with tab2:

    st.subheader(
        "📊 So sánh các mô hình Deep Learning"
    )

    st.info(
        "Các chỉ số Accuracy, Precision, Recall và F1 "
        "cần lấy từ kết quả đánh giá thực tế. "
        "Không sử dụng số liệu ước lượng."
    )

    # Thong tin kich thuoc checkpoint local
    model_rows = []

    for name, key in MODEL_OPTIONS.items():

        path_map = {
            "mobilenetv3":
                BASE_DIR /
                "mobilenet_model/mobilenet_v3_small_best.pth",

            "resnet18":
                BASE_DIR /
                "resnet_model/resnet18_best.pth",

            "custom_cnn":
                BASE_DIR /
                "cnn_model/optimized_custom_cnn_best.pth"
        }

        model_path = path_map[key]

        if model_path.exists():
            size_mb = (
                model_path.stat().st_size /
                (1024 * 1024)
            )
            size_text = f"{size_mb:.2f} MB"
        else:
            size_text = "Chưa có file"

        model_rows.append({
            "Mô hình": name,
            "Checkpoint": size_text,
            "Nguồn": (
                "DagsHub Registry + Local"
                if key == "mobilenetv3"
                else "Local"
            )
        })

    st.dataframe(
        pd.DataFrame(model_rows),
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.subheader(
        "📈 Biểu đồ huấn luyện Loss / Accuracy"
    )

    selected_chart = st.selectbox(
        "Chọn mô hình để xem biểu đồ",
        list(CHART_PATHS.keys())
    )

    chart_path = CHART_PATHS[selected_chart]

    if chart_path.exists():

        chart_image = Image.open(
            chart_path
        )

        st.image(
            chart_image,
            caption=(
                f"Training metrics - {selected_chart}"
            ),
            width="stretch"
        )

    else:
        st.warning(
            f"Không tìm thấy biểu đồ: {chart_path}"
        )

    st.divider()

    st.subheader("🔗 MLflow Tracking")

    st.markdown(
        "[Mở MLflow Experiments trên DagsHub]"
        "(https://dagshub.com/thaibao3214/"
        "smart-parking-mlops/experiments)"
    )

    st.markdown(
        "[Mở Model Registry trên DagsHub]"
        "(https://dagshub.com/thaibao3214/"
        "smart-parking-mlops/models)"
    )

    st.markdown("""
    **Thông tin hệ thống MLOps:**

    - DVC quản lý phiên bản dữ liệu.
    - MLflow Tracking lưu lịch sử huấn luyện.
    - MLflow Model Registry quản lý phiên bản model.
    - MobileNetV3-Small là model được chọn triển khai.
    - FastAPI phục vụ inference trên CPU.
    - Streamlit hiển thị kết quả nhận diện.
    """)


# =====================================================
# 10. FOOTER
# =====================================================

st.divider()

st.caption(
    "Smart Parking MLOps | "
    "PyTorch + FastAPI + Streamlit + MLflow + DagsHub + DVC"
)
