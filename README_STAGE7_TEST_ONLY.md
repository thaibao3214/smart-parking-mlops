# Stage 7 TEST ONLY — PKLot train/valid GPU integration

This **manually triggered**, distinct workflow verifies integration of GitHub Actions → DVC/DagsHub → private Kaggle dataset → Kaggle GPU (three models) → CPU evaluation → **separate** DagsHub MLflow experiment.

**NOT** new camera data or independent evaluation. No `approved_online/`, no `approved_online.dvc`, no test-derived 2,000 crops, no Production Registry updates.

Training: 32 Empty + 32 Occupied crops from `data/train`.
Evaluation: 128 Empty + 128 Occupied crops from `data/valid`.
Uses baseline v1 DVC checkpoints and 1 epoch per model. PKLot frame dependence still limits conclusions about generalization.

Files to add:
- `.github/workflows/stage7_test_only.yml`
- `pipeline/stage7_test_prepare.py`
- `pipeline/stage7_test_log.py`
- `kaggle/stage7_test_train.py`
- `README_STAGE7_TEST_ONLY.md`

Run workflow **Smart Parking - Stage 7 TEST ONLY GPU Integration** manually on `main`.

Successful run produces an artifact named `smart-parking-stage7-TEST-ONLY-...` and MLflow experiment `smart-parking-stage7-TEST-ONLY`. Do not use these checkpoints for production promotion.

This is an integration test, **not** evidence that `approved_online.dvc` push trigger or real-data retraining works. Those must be verified separately using genuinely new labeled camera data.
