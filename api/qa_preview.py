"""Read-only access to the reviewed MobileNetV3 QA candidate from one GitHub run.

Never creates model versions or changes MLflow aliases. Used only for explicit
preview requests; the production inference path remains untouched.
"""
from __future__ import annotations

import re

TARGET_RUN_ID = "45903240562"  # Successful Stage 7 test-only run for batch_005
VARIANT = "qa_preview"
REQUIRED_TAGS = {
    "github_run_id": TARGET_RUN_ID,
    "review_status": "QA_ONLY",
    "test_only": "true",
    "production_approved": "false",
    "data_origin": "pklot_train_simulation",
}


def find_qa_version(manager, model_name):
    """Return one exact QA model version or raise; never select 'latest'."""
    with manager.lock:
        if manager.client is None:
            manager.connect()
        versions = manager.client.search_model_versions(f"name = '{model_name}'")
        matches = []
        for version in versions:
            tags = dict(getattr(version, "tags", {}) or {})
            checksum = str(tags.get("stage7_candidate_sha256", ""))
            if (all(str(tags.get(key, "")) == value for key, value in REQUIRED_TAGS.items())
                    and re.fullmatch(r"[0-9a-fA-F]{64}", checksum)):
                matches.append(version)
        if len(matches) != 1:
            raise LookupError("Expected exactly one verified QA version for the selected run")
        row = matches[0]
        # A QA preview must not resolve to the model actively serving production.
        prod = manager.client.get_model_version_by_alias(model_name, "production")
        if str(row.version) == str(prod.version):
            raise RuntimeError("QA candidate is currently the production version")
        return row


def load_qa_model(manager, mlflow, torch, model_name):
    """Lazy-load QA weights into a separate cache; never replace manager.model."""
    with manager.lock:
        row = find_qa_version(manager, model_name)
        version = str(row.version)
        key = ("qa_preview", TARGET_RUN_ID, version)
        if key not in manager.other_models:
            model = mlflow.pytorch.load_model(
                f"models:/{model_name}/{version}", map_location="cpu"
            )
            if hasattr(model, "eval"):
                model.eval()
            with torch.inference_mode():
                output = model(torch.zeros(1, 3, 64, 64))
            if tuple(output.shape) != (1, 2):
                raise RuntimeError("Invalid QA model output; expected two classes")
            manager.other_models[key] = model
        return manager.other_models[key], version, "DagsHub QA (dùng thử)"
