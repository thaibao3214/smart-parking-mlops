# Smart Parking — Stage 5: MLflow + Evaluation Gate

## Files to copy into project root
- `.github/workflows/stage5_mlflow_gate.yml`
- `pipeline/stage5_evaluate.py`

Do **not** overwrite `pipeline/prepare_stage2.py` or `kaggle/stage4_two_models.py`.

## What it does
1. GitHub Actions downloads actual Stage 3 run `43458044027` and Stage 4 run `43479352874` artifacts.
2. Checks job completion, Git commit and DVC hash provenance, same 128 train/64 validation subsets, no online/test crops, and SHA256 of three candidate checkpoints.
3. Compares each candidate's **reported** validation accuracy against corresponding v1, uses 98% minimal accuracy gate and zero allowed regression. This is an **audit of existing Stage 3/4 reports**, **not a new inference or evaluation on a larger set**.
4. Logs 3 model runs plus a gate-summary run to DagsHub MLflow experiment `smart-parking-stage5-evaluation`.
5. Uploads `stage5_report.json` and `stage5_summary.md` to GitHub Actions artifacts. **No model registration or Production alias change occurs.**

## GitHub setup
- Existing secret: `DAGSHUB_TOKEN` (needs write access to DagsHub MLflow).
- Workflow uses `GITHUB_TOKEN` automatically for artifact download (permission `actions: read`).
- If you run it >90 days later, GitHub artifacts may have expired and stage3/stage4 downloads will fail; rebuild stage5 input artifacts or update workflow.

## Setup
From Windows PowerShell in `C:\Users\thaib\Downloads\project`:

```powershell
git add .github/workflows/stage5_mlflow_gate.yml
git add pipeline/stage5_evaluate.py
git diff --cached --name-only
git commit -m "feat: Stage 5 MLflow audit and safe evaluation gate"
git push github main
```

GitHub Actions → **Smart Parking - Stage 5 MLflow Evaluation Gate** → Run workflow (main).

Expected logs:
```
STAGE5_ARTIFACT_DOWNLOAD_PASS
STAGE5_ARTIFACT_INTEGRITY_PASS
STAGE5_IDENTICAL_VALIDATION_SPLIT_PASS
STAGE5_THREE_CANDIDATES_CHECKSUM_PASS
STAGE5_GATE: PROVISIONAL_PASS_PRODUCTION_BLOCKED
STAGE5_PRODUCTION_UNCHANGED_PASS
STAGE5_MLFLOW_TRACKING_PASS
STAGE5_END_TO_END_PASS
```

## Local audit without credentials/network
For downloaded ZIP files:

```powershell
python pipeline/stage5_evaluate.py --stage3 smart-parking-stage3-43458044027-1.zip --stage4 smart-parking-stage4-43479352874-1.zip --offline
```

## Limitations / next engineering steps
- Stage 3 and 4 reported 63/64 correct for each model, but no per-sample labels/predictions or timing, so F1, confusion matrices and CPU latency **cannot be inferred**. Do not claim production-ready accuracy from this small validation set.
- The pipeline does not upload large `.pth` to MLflow or create Registry model versions; it links lineage with SHA256 and original GitHub run IDs.
- Before enabling deployment, evaluate on properly independent data and measure F1, per-class recall and CPU latency; then add reviewed production-approval/rollback workflow.
