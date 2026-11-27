"""Vietnamese Streamlit selector for a read-only MLflow QA preview."""
import requests
import streamlit as st

TARGET_RUN_ID = "45903240562"


@st.cache_data(ttl=30, show_spinner=False)
def _get_qa_availability(api_base_url):
    try:
        response = requests.get(f"{api_base_url}/qa-model-info", timeout=10)
        response.raise_for_status()
        info = response.json()
        if (info.get("available") is True
                and info.get("review_status") == "QA_ONLY"
                and str(info.get("github_run_id")) == TARGET_RUN_ID
                and info.get("test_only") is True
                and str(info.get("version", "")).isdigit()):
            return info
    except (requests.RequestException, ValueError):
        pass
    return None


def render_qa_selector(api_base_url, model_type):
    """Return an explicit preview variant only if FastAPI verified QA provenance."""
    if model_type != "mobilenetv3":
        return "production"

    qa = _get_qa_availability(api_base_url)
    reviewed = st.session_state.get("qa_review_approved")
    permitted = (qa is not None and isinstance(reviewed, dict)
                 and reviewed.get("version") == str(qa["version"])
                 and reviewed.get("run_id") == str(qa["github_run_id"]))
    labels = ["Mô hình đang sử dụng"]
    if permitted:
        labels.append("Mô hình vừa huấn luyện (đã chốt dùng thử)")
    # A new widget key is used when approval changes, avoiding stale radio values.
    widget_key = f"qa_preview_variant_v3_{qa['version'] if permitted else 'locked'}"
    selected = st.radio("Phiên bản mô hình", labels, index=1 if permitted else 0,
                        key=widget_key)

    if qa is None:
        st.caption("Chưa tìm thấy mô hình QA hợp lệ trên DagsHub Registry.")
        return "production"
    if not permitted:
        st.info("Đã có mô hình vừa huấn luyện. Vào **Huấn luyện & theo dõi** để xem "
                "chỉ số và **chốt dùng thử** trước khi nhận diện bằng mô hình mới.")
        return "production"
    if selected == labels[-1] and len(labels) == 2:
        st.info(f"Đang dùng thử phiên bản {qa['version']}. "
                "Mô hình Production không thay đổi.")
        return "qa_preview"
    return "production"
