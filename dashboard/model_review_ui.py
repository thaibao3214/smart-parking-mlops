"""Read-only Stage 7 evaluation and explicit, session-only QA model review.

Neither this module nor its buttons modify MLflow aliases or Production.
The Github JSON report must belong to the QA candidate's exact run.
"""
from __future__ import annotations

import io
import os
import zipfile

import pandas as pd
import requests
import streamlit as st

from qa_preview_ui import _get_qa_availability, TARGET_RUN_ID

REPO = "thaibao3214/smart-parking-mlops"
DISPLAY_NAMES = {
    "mobilenetv3": "MobileNetV3",
    "resnet18": "ResNet18",
    "custom_cnn": "Custom CNN",
}


@st.cache_data(ttl=120, show_spinner=False)
def _read_evaluation_report(run_id: str):
    """Fetch the small GitHub evaluation report for this exact workflow run."""
    token = os.environ.get("GH_TOKEN", "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "smart-parking-readonly-review",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    base = f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/artifacts"
    response = requests.get(base, headers=headers, params={"per_page": 100}, timeout=12)
    response.raise_for_status()
    artifacts = response.json().get("artifacts", [])
    prefix = f"smart-parking-stage7-REPORT-{run_id}-"
    candidates = [
        a for a in artifacts
        if str(a.get("name", "")).startswith(prefix)
        and not a.get("expired", False)
        and a.get("archive_download_url")
    ]
    if not candidates:
        return None

    # Choose the newest attempt for this SAME run.
    candidates.sort(key=lambda a: int(a.get("id") or 0), reverse=True)
    if int(candidates[0].get("size_in_bytes") or 0) > 5_000_000:
        raise ValueError("Báo cáo quá lớn; từ chối đọc")
    archive = requests.get(
        candidates[0]["archive_download_url"], headers=headers, timeout=20
    )
    archive.raise_for_status()
    if len(archive.content) > 5_000_000:
        raise ValueError("Gói báo cáo vượt giới hạn kích thước")
    with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
        matching = [
            name for name in z.namelist()
            if name.endswith("stage7_test_report.json")
            and not name.startswith("/")
            and ".." not in name.split("/")
        ]
        if len(matching) != 1 or z.getinfo(matching[0]).file_size > 2_000_000:
            raise ValueError("Không có báo cáo Evaluation Gate hợp lệ")
        import json
        report = json.loads(z.read(matching[0]).decode("utf-8"))
    if (str(report.get("github_run_id")) != str(run_id)
            or report.get("test_only") is not True
            or report.get("production_approved") is not False
            or report.get("registry_alias_changed") is not False):
        raise ValueError("Báo cáo không khớp run / hợp đồng TEST ONLY")
    return report


def _score(value):
    try:
        n = float(value)
        if 0 <= n <= 1:
            return f"{n * 100:.2f}%"
    except (ValueError, TypeError):
        pass
    return "—"


def render_model_review(api_base_url: str):
    """Explicit QA trial decision. No Production deployment or Registry write."""
    st.subheader("✅ Đánh giá và chọn mô hình sau huấn luyện")
    st.caption("Chỉ chốt phiên bản **dùng thử** trên trang nhận diện; không thay mô hình đang phục vụ.")

    qa = _get_qa_availability(api_base_url)
    if qa is None:
        st.warning(
            "Chưa tìm thấy phiên bản mô hình mới đã được xác minh trên DagsHub. "
            "Mô hình đang sử dụng vẫn được giữ nguyên."
        )
        return

    model_name = "MobileNetV3"
    version = str(qa["version"])
    run_id = str(qa["github_run_id"])
    st.success(f"Mô hình được đề xuất để xem xét: **{model_name} — phiên bản {version}**")

    report = None
    try:
        report = _read_evaluation_report(run_id)
    except (requests.RequestException, ValueError, zipfile.BadZipFile) as err:
        with st.expander("Không tải được bảng đánh giá đầy đủ từ GitHub"):
            st.caption(f"{type(err).__name__}: {str(err)[:180]}")

    if report is not None:
        rows = []
        for key, label in DISPLAY_NAMES.items():
            row = (report.get("models") or {}).get(key) or {}
            candidate = row.get("candidate") or {}
            latency = row.get("latency") or {}
            try:
                p95 = float(latency.get("cpu_forward_p95_ms"))
                p95_str = f"{p95:.2f} ms"
            except (TypeError, ValueError):
                p95_str = "—"
            rows.append({
                "Mô hình": label,
                "Macro F1 (validation)": _score(candidate.get("macro_f1")),
                "CPU p95": p95_str,
                "Đánh giá": "Đạt" if row.get("gate_pass") is True else "Không đạt",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        winner = report.get("suggested_for_review")
        if winner != "mobilenetv3":
            st.error("Mô hình QA trên Registry không khớp model được đề xuất của báo cáo. Không cho chốt.")
            return
    else:
        st.caption("Không lấy được bảng của cả ba mô hình. Vẫn có thể xem chỉ số của mô hình QA đã đăng ký.")

    f1 = qa.get("candidate_macro_f1")
    accuracy = qa.get("candidate_accuracy")
    c1, c2 = st.columns(2)
    c1.metric("Macro F1 trên tập validation", _score(f1))
    c2.metric("Accuracy trên tập validation", _score(accuracy))
    st.caption(
        "Các tỷ lệ này là kết quả đánh giá trên tập validation PKLot, "
        "không phải xác suất đúng của một ảnh mới."
    )

    current = st.session_state.get("qa_review_approved")
    chosen = isinstance(current, dict) and current.get("version") == version and current.get("run_id") == run_id
    if chosen:
        st.success("Đã chốt **dùng thử** mô hình mới trong phiên làm việc này. Vào **Thử nhận diện** để xem kết quả.")
        if st.button("Quay lại mô hình đang sử dụng", key="qa_review_reset"):
            st.session_state.pop("qa_review_approved", None)
            st.rerun()
    else:
        confirm = st.checkbox(
            "Tôi xác nhận chỉ chọn mô hình mới để dùng thử, không triển khai chính thức.",
            key="qa_review_confirm",
        )
        if st.button("Chốt mô hình đề xuất để dùng thử", type="primary", disabled=not confirm,
                     key="qa_review_accept"):
            st.session_state["qa_review_approved"] = {"version": version, "run_id": run_id}
            st.rerun()

    st.caption(
        "Để thay Production cần bước phê duyệt và workflow triển khai riêng, "
        "có kiểm tra và phương án rollback. Lựa chọn ở đây không đổi MLflow alias."
    )
