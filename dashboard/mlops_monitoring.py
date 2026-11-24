"""MLOps monitoring and protected local Stage 7 TEST_ONLY control.

Sources: MLflow Registry/Tracking (live), public GitHub Actions API (live),
and local immutable Stage 7 batch markers (DVC metadata only).
Only Stage 7 control module may launch the existing pipeline; no Production promotion.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

REPOSITORY = "thaibao3214/smart-parking-mlops"
MODEL_NAME = "SmartParking-MobileNetV3-Small"
DEFAULT_TRACKING_URI = f"https://dagshub.com/{REPOSITORY}.mlflow"
WORKFLOW = "stage7_test_only.yml"
GITHUB = f"https://github.com/{REPOSITORY}"


def _time(iso):
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return str(iso)


def _local_batches(root):
    """Only consume batch markers, never open or modify training data."""
    rows = []
    directory = root / "pipeline" / "stage7_batches"
    for file in sorted(directory.glob("*.json")) if directory.exists() else []:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("batch_id") != file.stem:
                continue
            if data.get("mode") != "TEST_ONLY_NO_PROMOTION":
                continue
            counts = data.get("class_counts") or {}
            rows.append({
                "Batch": data.get("batch_name", "—"),
                "Batch ID": file.stem,
                "Frames": data.get("frames"),
                "Crops": data.get("crops"),
                "Empty": counts.get("Empty"),
                "Occupied": counts.get("Occupied"),
                "Excluded IDs": ", ".join(map(str, data.get("excluded_source_image_ids") or [])) or "—",
                "Mode": "TEST_ONLY",
                "DVC pointer": data.get("confirmed_pointer", "—"),
            })
        except (OSError, ValueError, TypeError):
            continue
    return rows


@st.cache_data(ttl=120, show_spinner=False)
def _github_runs():
    response = requests.get(
        f"https://api.github.com/repos/{REPOSITORY}/actions/workflows/{WORKFLOW}/runs",
        params={"per_page": 12},
        headers={"Accept": "application/vnd.github+json", "User-Agent": "smart-parking-monitor"},
        timeout=10,
    )
    response.raise_for_status()
    return [
        {
            "Run": run.get("run_number"),
            "Trạng thái": run.get("conclusion") or run.get("status") or "unknown",
            "Sự kiện": run.get("event") or "—",
            "Commit": (run.get("head_sha") or "")[:8],
            "Thời gian": _time(run.get("created_at")),
            "URL": run.get("html_url"),
        }
        for run in response.json().get("workflow_runs", [])
    ]


@st.cache_data(ttl=120, show_spinner=False)
def _mlflow_snapshot(uri):
    """Fetch real Registry/Tracking values; never manufacture a model version."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    versions = list(client.search_model_versions(f"name = '{MODEL_NAME}'"))
    versions.sort(key=lambda x: int(x.version), reverse=True)
    production = None
    try:
        production = str(client.get_model_version_by_alias(MODEL_NAME, "Production").version)
    except Exception:
        # Absence of alias / unavailable registry must not be treated as v1.
        pass
    # Existing QA versions may not have an explicit QA alias. Show latest as candidate.
    latest = str(versions[0].version) if versions else None
    rows = []
    for v in versions[:12]:
        details = {
            "Version": str(v.version),
            "Vai trò": "Production" if str(v.version) == production else "Candidate / historical",
            "Run ID": v.run_id or "—",
            "Trạng thái registry": str(v.status or "—"),
            "Source": str(v.source or "—"),
            "Metrics": {},
            "Datasets": [],
            "Experiment ID": None,
        }
        if v.run_id:
            try:
                run = client.get_run(v.run_id)
                details["Experiment ID"] = run.info.experiment_id
                details["Metrics"] = {str(k): float(x) for k, x in run.data.metrics.items()}
                inputs = getattr(run, "inputs", None)
                for entry in (getattr(inputs, "dataset_inputs", None) or []):
                    ds = getattr(entry, "dataset", None)
                    if ds is not None:
                        details["Datasets"].append({
                            "Tên": getattr(ds, "name", "—"),
                            "Digest": getattr(ds, "digest", "—"),
                        })
            except Exception as err:
                details["Read error"] = str(err)[:250]
        rows.append(details)
    return {"production": production, "latest": latest, "versions": rows}


def _metric_details(metrics):
    # No guessing metric semantics: display original MLflow names and values.
    return pd.DataFrame([
        {"MLflow metric": k, "Giá trị": round(v, 6)}
        for k, v in sorted(metrics.items())
    ])


def render_mlops_monitoring(root: Path, api_health=None):
    st.subheader("🛠️ MLOps Monitoring")
    st.caption(
        "Chỉ đọc dữ liệu thực tế từ MLflow, GitHub Actions và batch marker DVC trên máy. "
        "Các phiên PKLot train là mô phỏng TEST_ONLY, không phải camera ngoài thực tế."
    )
    if st.button("🔄 Làm mới dữ liệu MLOps", key="mlops_refresh"):
        _github_runs.clear()
        _mlflow_snapshot.clear()
        st.rerun()

    st.markdown("#### Model Registry & MLflow Tracking")
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI).rstrip("/")
    if uri.startswith("http://") and not uri.startswith("http://localhost") and not uri.startswith("http://127.0.0.1"):
        st.error("MLflow URI qua mạng phải dùng HTTPS.")
        return

    try:
        registry = _mlflow_snapshot(uri)
    except ImportError:
        st.warning("Chưa cài MLflow trong môi trường Streamlit: `pip install mlflow`.")
        registry = None
    except Exception as err:
        st.warning("Chưa lấy được Registry MLflow. Kiểm tra kết nối và biến môi trường MLFLOW_TRACKING_*. ")
        with st.expander("Chi tiết kết nối MLflow"):
            st.code(f"{type(err).__name__}: {str(err)[:300]}")
        registry = None

    if registry is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Production alias", "v" + registry["production"] if registry["production"] else "Chưa xác minh")
        c2.metric("Phiên bản Registry mới nhất", "v" + registry["latest"] if registry["latest"] else "—")
        c3.metric("Số phiên bản đang hiển thị", len(registry["versions"]))
        if api_health:
            st.caption(
                "FastAPI hiện báo: "
                f"source={api_health.get('model_source', 'Unknown')} · "
                f"version={api_health.get('model_version', 'Unknown')} · "
                f"alias={api_health.get('registry_alias', 'Unknown')}"
            )
        if registry["versions"]:
            versions = registry["versions"]
            st.dataframe(pd.DataFrame([{
                "Version": v["Version"], "Vai trò": v["Vai trò"],
                "Run ID": v["Run ID"], "Trạng thái": v["Trạng thái registry"]
            } for v in versions]), hide_index=True, width="stretch")
            selected = st.selectbox(
                "Xem metrics và lineage của phiên bản",
                [v["Version"] for v in versions],
                key="mlops_version"
            )
            v = next(x for x in versions if x["Version"] == selected)
            if v["Metrics"]:
                st.dataframe(_metric_details(v["Metrics"]), hide_index=True, width="stretch")
            else:
                st.info("Không có metrics đọc được cho phiên bản này.")
            if v["Datasets"]:
                st.write("Datasets ghi nhận trong MLflow Run:")
                st.dataframe(pd.DataFrame(v["Datasets"]), hide_index=True, width="stretch")
            else:
                st.caption("Chưa lấy được lineage dataset từ MLflow Run này.")
            if v["Run ID"] != "—" and v["Experiment ID"] is not None:
                st.link_button(
                    "Mở MLflow Run",
                    f"https://dagshub.com/{REPOSITORY}.mlflow/#/experiments/{v['Experiment ID']}/runs/{v['Run ID']}",
                )
        st.link_button("Mở DagsHub MLflow", f"https://dagshub.com/{REPOSITORY}.mlflow")

    st.divider()
    st.markdown("#### Stage 7 — Batch history & DVC")
    markers = _local_batches(Path(root))
    if markers:
        st.dataframe(pd.DataFrame(markers), hide_index=True, width="stretch")
        st.caption("Chỉ số liệu trong marker đã ghi nhận; không đồng nghĩa đã deploy Production.")
    else:
        st.info("Chưa tìm thấy batch marker tại `pipeline/stage7_batches/` trên máy này.")

    st.divider()
    st.markdown("#### GitHub Actions — Stage 7 TEST_ONLY")
    try:
        runs = _github_runs()
        if runs:
            st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch",
                         column_config={"URL": st.column_config.LinkColumn("Mở run")})
            st.caption("Danh sách các run gần đây của workflow, không suy diễn run thuộc batch nào.")
        else:
            st.info("Chưa có GitHub Actions run công khai.")
    except requests.RequestException as err:
        st.warning(f"Không đọc được GitHub Actions: {type(err).__name__}.")
    st.link_button("Mở GitHub Actions", f"{GITHUB}/actions/workflows/{WORKFLOW}")
    from stage7_control import render_stage7_control
    render_stage7_control(Path(root))
