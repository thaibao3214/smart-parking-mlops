# Stage 7 — Continuous Training / MLflow Gate / Manual Production Promotion

## Goal and safety boundary

When **new, externally captured parking-space crops** have been **labeled and explicitly approved** by a human, commit a new DVC pointer. GitHub Actions then pulls those exact files, uploads a private per-run Kaggle Dataset, runs 1-epoch GPU fine-tuning for **MobileNetV3, ResNet18, Custom CNN**, measures metrics and CPU forward latency, and logs review results to DagsHub MLflow.

- Automatically triggered by **a Git push modifying `approved_online.dvc` on `main`**; manual `workflow_dispatch` is also supported.
- **Never uses `data/test` or `data/realtime_stream` as training data.** The original 2,000 simulated online crops originated in test and are **not eligible**.
- Evaluation uses **PKLot `data/valid`**, 128 crops per class; this is not an untouched external blind test.
- GitHub Actions **never changes `production` alias**, never pushes candidate checkpoints back into DVC automatically, and does not silently update the FastAPI app.
- A separate manual tool can register an approved MobileNetV3 PyTorch model in MLflow and switch the alias, storing the old version number for rollback.
- **Do not run the CT pipeline until you have truly new, separately captured and reviewed images.** A synthetic or PKLot-test-derived manifest violates the provenance requirement.

## File placement

```
project/
  .github/workflows/stage7_continuous_training.yml
  pipeline/stage7_prepare.py
  pipeline/stage7_gate.py
  pipeline/stage7_promote.py
  kaggle/stage7_train.py
  README_STAGE7.md
```

The new `approved_online/` folder is intentionally **outside `data/`**: `data.dvc` already tracks the full `data/` directory, and nested DVC tracking would fail.

The project must also retain `retraining/evaluate.py`, `data.dvc`, and DVC pointers for all three v1 checkpoints. Required GitHub Secrets: `DAGSHUB_TOKEN`, `KAGGLE_API_TOKEN` (already used in Stage 1–6).

## 1 — Prepare independently captured *new* data

Create a local folder `approved_online/images/` with manually labeled cropped images from **a real new camera session**, not PKLot original train/valid/test or the `realtime_stream` simulation. Crop filenames can be unique; model input transform converts each crop to 64×64 RGB.

Create `approved_online/manifest.json` with schema:

```json
{
  "schema_version": 1,
  "origin": "external_camera_new_capture",
  "source_split": "external_new_capture",
  "contains_pklot_test": false,
  "approved": true,
  "approved_by": "human-reviewer-identifier",
  "approved_at": "2026-11-14T12:00:00+07:00",
  "samples": [
    {
      "file": "images/new-frame001-space01.png",
      "label": 0,
      "reviewed": true,
      "source_frame_id": "new-camera-frame001",
      "space_id": "space01",
      "sha256": "<actual 64-character hex digest of image file>"
    }
  ]
}
```

The sample shown is a **schema example**, not enough data to train. Provide **at least 16 Empty (`label: 0`) + 16 Occupied (`label: 1`)** approved crops. Many parking spaces from the same frame may be included if `space_id` differs. Each `(source_frame_id, space_id)` must be unique. The `sha256` must be calculated from the exact image bytes. The `approved_by`, `approved_at`, `origin`, and `reviewed` fields represent actual human review; scripts cannot independently prove camera provenance.

Example PowerShell to calculate the digest of a file:

```powershell
(Get-FileHash approved_online/images/new-frame001-space01.png -Algorithm SHA256).Hash.ToLower()
```

**Do not add `approved_online` images directly to Git.** DVC will store the actual files remotely; only its `.dvc` pointer goes to GitHub.

## 2 — Register the approved version in DVC

```powershell
cd C:\Users\thaib\Downloads\project
python -m py_compile pipeline/stage7_prepare.py pipeline/stage7_gate.py pipeline/stage7_promote.py kaggle/stage7_train.py
dvc add approved_online
dvc push approved_online.dvc -r origin
git add .github/workflows/stage7_continuous_training.yml
git add pipeline/stage7_prepare.py pipeline/stage7_gate.py pipeline/stage7_promote.py
git add kaggle/stage7_train.py README_STAGE7.md
git add approved_online.dvc
# If DVC updated the root .gitignore, also stage it:
# git add .gitignore
git diff --cached --name-only
git commit -m "feat: Stage 7 continuous training with reviewed online data"
git push github main
```

A push changing `approved_online.dvc` triggers Stage 7 automatically. If Stage 7 code is committed before your first eligible data batch, just omit the `dvc add` / `dvc push` / pointer `git add` commands and commit only the code; run Stage 7 later when real approved data exists.

If `.gitignore` ignores DVC pointers, Git may require:

```powershell
git add -f approved_online.dvc
```

Never use `git add .` for this project. Verify the staged list contains **no secret or raw image** files.

## 3 — Check GitHub Actions

Workflow name: **Smart Parking - Stage 7 Continuous Training (review only)**.

Expected log markers:

```
STAGE7_DVC_PULL_PASS
STAGE7_HUMAN_APPROVAL_VALIDATED_PASS
STAGE7_PREPARE_PASS
STAGE7_KAGGLE_DATASET_READY_PASS
STAGE7_KAGGLE_PUSH_PASS
STAGE7_THREE_CANDIDATES_RECEIVED_PASS
STAGE7_EVALUATION_GATE_PASS
STAGE7_PRODUCTION_UNCHANGED_PASS
STAGE7_MLFLOW_TRACKING_PASS
STAGE7_END_TO_END_PASS
STAGE7_WORKFLOW_PASS
```

Artifact: `smart-parking-stage7-<run_id>-<attempt>`. Contains three candidate `.pth` files, train report, evaluation report, and MLflow audit summary. A Gate `FAIL` for an individual candidate is a **legitimate model-quality result**; it does not automatically fail workflow execution.

Validation F1/Accuracy and CPU latency are **not** sufficient on their own to establish safety or effectiveness on an unseen real camera. Review image examples, data provenance and drift before production changes.

## 4 — Manual Production promotion and rollback (optional, after separate review)

Your current `main.py` loads **`SmartParking-MobileNetV3-Small@production`** from DagsHub Registry. To promote a Stage 7 MobileNetV3 candidate, an operator needs to independently verify its report, data provenance and Gate result, then download the Stage 7 artifact to the local project machine and unzip it.

On that local machine, with installed `torch`, `torchvision` and `mlflow==2.22.2`, provide `DAGSHUB_TOKEN` securely as an environment variable. **Never place a token into a Git-tracked script.**

```powershell
python pipeline/stage7_promote.py `
  --report stage7_output/stage7_report.json `
  --candidate stage7_logs/stage7_candidate_mobilenetv3.pth `
  --approve-run-id <ACTUAL_RUN_ID> `
  --approve-sha256 <ACTUAL_CHECKPOINT_SHA256> `
  --confirm-production
```

The tool checks source run, exact checkpoint SHA256, Gate, Registry model compatibility, and tests a freshly registered version before moving the `production` alias. It prints the old version for rollback:

```powershell
python pipeline/stage7_promote.py --rollback-version <OLD_VERSION> --confirm-production
```

**Be aware:** changing the DagsHub alias does not guarantee the local FastAPI has reloaded immediately. Restart or invoke its reload endpoint and test inference. Its registry model currently targets MobileNetV3; a ResNet18 or Custom CNN production switch requires separate, explicit API changes.

## Remaining acceptance tests

1. Real externally captured samples are human-labeled and approved, stored with DVC and pulled at the exact Git revision.
2. A new commit of the approved `.dvc` pointer triggers Stage 7 **without a manual Run workflow click**.
3. Kaggle processes the exact current-run private Dataset; all three candidates and SHA checks pass.
4. The MLflow experiment contains three runs and metrics; only models meeting the Gate are suggested.
5. Test manual Registry promotion, FastAPI reload/inference and rollback. **No automatic alias changes are permitted.**
