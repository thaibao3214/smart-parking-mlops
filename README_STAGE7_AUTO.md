# Stage 7 — automatic TEST_ONLY trigger and QA registration

This update **does not** change official `pipeline/stage7_prepare.py`, real-data approval, FastAPI Production alias, or the previous evaluation gate. It runs only on **20 manually confirmed PKLot train simulation frames**, as a technical test; the holdout is PKLot valid and is not an independent final generalization estimate.

## 1. Install files

Unzip into `C:\Users\thaib\Downloads\project`, replacing only `.github/workflows/stage7_test_only.yml` and adding:

- `pipeline/stage7_auto_trigger.py`
- `pipeline/stage7_auto_register_qa.py`
- `README_STAGE7_AUTO.md`

No need to replace the previously working training, data preparation, or logging scripts.

## 2. Publish automation code ONE time

```powershell
cd C:\Users\thaib\Downloads\project
python -m py_compile pipeline\stage7_auto_trigger.py pipeline\stage7_auto_register_qa.py

git add pipeline/stage7_auto_trigger.py pipeline/stage7_auto_register_qa.py .github/workflows/stage7_test_only.yml README_STAGE7_AUTO.md
git diff --cached --stat
git commit -m "Automate Stage7 confirmed-batch trigger and QA registry"
git push github HEAD:main
```

**Do not use `git add .`.** Remaining edits to API/dashboard/.gitignore are separate; commit them intentionally only after review.

## 3. One-time automatic run for your existing 20-frame confirmed batch

Your 20-frame/2,000-crop batch is already confirmed, so no need to review again.

```powershell
python pipeline\stage7_auto_trigger.py --once --dry-run
python pipeline\stage7_auto_trigger.py --once
```

The second command automatically validates the batch (including every 64x64 crop), creates a deterministic batch ID, runs `dvc add` / `dvc push`, commits **only** DVC pointers and `pipeline/stage7_batches/<batch_id>.json`, and pushes to GitHub. The GitHub push triggers Stage 7 without clicking `Run workflow`.

Requirements: existing `dvc` credentials for DagsHub and `git push github` authentication must work. The bridge **runs as a local process**; it is not an always-on cloud service. It does not store tokens in Streamlit or the Git repository.

## 4. Future camera batches

After a new batch has been collected and confirmed, run the watcher locally in a separate terminal:

```powershell
python pipeline\stage7_auto_trigger.py --watch
```

It checks for a complete batch every 15 seconds. It will **not** retrigger the same batch. The current simulation UI uses a fixed directory and prevents overwriting saved confirmations; to collect another batch, create a new simulation session with clearly separated storage before reusing this approach. This watcher supports the existing one fixed `confirmed_train_sim` path only.

The GitHub workflow proceeds automatically to DVC pull, Kaggle training, Evaluation Gate, MLflow Tracking, **QA registration**. Registration is dynamic: MobileNetV3 / ResNet18 / Custom CNN, whichever was selected by the gate. It verifies the run/commit, training-provenance manifest, **all three model checkpoint SHA256 hashes**, and the F1+latency ranking. It tags a QA-only registry version and **does not set/delete or update any alias**. The Production alias stays unchanged.

If no candidate passes the gate, QA registration is skipped.

## 5. Verification and limitations

Look for these log entries:

```
STAGE7_AUTO_TRIGGER_PUSH_PASS
STAGE7_TEST_ONLY_END_TO_END_PASS
STAGE7_AUTO_QA_VERIFICATION_PASS
STAGE7_QA_VERSION_REGISTERED (or STAGE7_QA_ALREADY_REGISTERED)
STAGE7_PRODUCTION_ALIAS_UNCHANGED
```

For registered-version lookup, review the MLflow Model Registry. There is no Production deployment in this TEST_ONLY flow.

If a Git push or DVC push fails, do **not** delete confirmed crop data or change the registered `production` alias. Inspect `git status` and the workflow run; a locally created batch marker might require an explicit `git push github HEAD:main` after fixing credentials.

GitHub Actions Git credentials are not used by Streamlit; the local watcher needs to stay running or be started explicitly. A fully unattended cloud capture architecture would require an always-on ingest service and a durable job queue, which are not part of this demo.
