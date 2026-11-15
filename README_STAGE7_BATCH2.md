# Stage 7 TEST_ONLY — Immutable batch 002

## What this changes

Maintains batch 1 untouched (`simulation_train`, `confirmed_train_sim`). Adds batch 2 in independent folders: `simulation_batches/batch_002` and `confirmed_batches/batch_002`. Changes the camera selector, API's filename/sha256 ROI lookup, Stage7 data preparation, DVC auto-trigger, workflow's batch selection, and QA dataset pointer tagging. **Does not change Production.**

The `prepare_train_simulation.py` in this package **replaces** the previous one; its default is now `batch_002`, not legacy `simulation_train`. It selects twenty previously unused PKLot-train frames sharing a layout with **100 parking spaces per frame**. If no such group remains, it fails instead of silently choosing a different or reused image.

## One-time installation

Copy the package file tree into the project. Preserve the existing files not listed here, especially `kaggle/stage7_test_train.py`, `pipeline/stage7_gate.py`, and `pipeline/stage7_test_log.py`.

Before committing, inspect the changed paths. Do NOT run `git add .` because this project has other untracked test artifacts.

```powershell
cd C:\Users\thaib\Downloads\project
python -m py_compile prepare_train_simulation.py api/main.py dashboard/online_stream.py pipeline/stage7_auto_trigger.py pipeline/stage7_test_prepare.py pipeline/stage7_auto_register_qa.py
python prepare_train_simulation.py --batch batch_002 --count 20 --seed 2026
```

Restart the FastAPI (`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`) **after** batch 2 creation so the backend reads its new manifest. Restart Streamlit (`python -m streamlit run dashboard/app.py`). The dashboard should now list `PKLot train — batch_002 (TEST_ONLY)`. Choose MobileNetV3 and manually verify each frame before saving. Do not copy labels from COCO.

After confirming 20 frames, check:

```powershell
(Get-ChildItem confirmed_batches\batch_002\frames -Filter labels.json -Recurse -File).Count
python pipeline\stage7_auto_trigger.py --once --batch batch_002 --dry-run
```

## Publish updated workflow (once)

**Commit workflow/code changes before triggering a batch.** In particular the new workflow must already be on GitHub main. Otherwise GitHub Actions may run the *old fixed-directory* pipeline.

```powershell
git add prepare_train_simulation.py api/main.py dashboard/online_stream.py pipeline/stage7_auto_trigger.py pipeline/stage7_test_prepare.py pipeline/stage7_auto_register_qa.py .github/workflows/stage7_test_only.yml README_STAGE7_BATCH2.md
git diff --cached --stat
git commit -m "Add Stage7 immutable multi-batch TEST_ONLY support"
git push github HEAD:main
```

Then run an **external/local watcher** in a separate terminal (watcher stays open while collection happens), or manually trigger once after confirmation:

```powershell
python pipeline\stage7_auto_trigger.py --once --batch batch_002
# OR, before collecting labels:
python pipeline\stage7_auto_trigger.py --watch --batch batch_002
```

The local watcher (not Streamlit itself) detects completion, versions both directories with DVC, commits two pointers plus exactly one batch marker, and pushes GitHub. GitHub Actions extracts that marker from the triggering commit and validates both `.dvc` hashes against the marker **before** packaging crops. If preparation fails, it does **not** silently fall back to batch 1.

## Limitations

- This is a PKLot training-set **simulation**. It demonstrates MLOps automation, not adaptation to independent real cameras.
- The 20 frames must each have 100 valid slots because current Stage7 QA registration/gate expects exactly 2,000 confirmed crops. The local trigger fails closed otherwise.
- Current GitHub workflow has an additional `workflow_dispatch` action that still targets the legacy batch 1. For batch 2 use the automatic push marker, not manual dispatch.
- Uses a single shared Kaggle kernel with workflow concurrency serialization. For a higher-throughput production system, use dedicated job IDs/queues.
- After `dvc add` the root `.gitignore` may gain two new ignore entries; review it before committing separately. Existing modifications to `.gitignore` are not silently included in auto commits.
