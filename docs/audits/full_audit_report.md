# Smart Parking MLOps — Full Project Audit

- Time: 2026-11-24T06:50:01+07:00
- Root: `C:\Users\thaib\Downloads\project`
- Mode: Read-only static evidence audit, not a functional end-to-end test
- Files scanned: 22453; Python files: 76

## Status by workstream

| Area | Status | Evidence |
|---|---|---|
| 01 - Dataset / COCO | CO_THANH_PHAN | 3/3 |
| 02 - Data versioning (DVC) | CO_THANH_PHAN | 3/3 |
| 03 - Model implementations | MOT_PHAN | 3/4 |
| 04 - MLflow / Registry / DagsHub | CO_THANH_PHAN | 2/2 |
| 05 - API / Inference | CO_THANH_PHAN | 2/2 |
| 06 - Streamlit dashboard | CO_THANH_PHAN | 3/3 |
| 07 - Online streaming / feedback | CO_THANH_PHAN | 2/2 |
| 08 - Retraining | CO_THANH_PHAN | 3/3 |
| 09 - Stage 5 and 6 evaluation | CO_THANH_PHAN | 3/3 |
| 10 - Stage 7 continuous training / gate | CO_THANH_PHAN | 5/5 |
| 11 - Stage 7 UI controls and QA | CO_THANH_PHAN | 4/4 |
| 12 - CI/CD GitHub Actions | CO_THANH_PHAN | 2/2 |
| 13 - Tests and documentation | CO_THANH_PHAN | 3/3 |

**Important:** CO_THANH_PHAN = files/signals exist, NOT that training, deployment or any workflow passed.

## Missing or incomplete signals

- 03 - Model implementations: Model-related Python files

## Python syntax

- Parsed: 76/76
- Syntax errors: 0

## Streamlit → retraining integration clues (NOT functional proof)

- dashboard_retraining_related_words: FOUND
- dashboard_launches_process_or_workflow: FOUND
- retrain_script_defines_training_operations: FOUND
- auto_trigger_contains_dispatch_related_code: FOUND

## COCO examples (sampled, metadata only)

- `data/realtime_stream/_annotations.coco.json`: readable; images=20; annotations=2000
- `data/test/_annotations.coco.json`: skipped_large; images=n/a; annotations=n/a
- `data/train/_annotations.coco.json`: skipped_large; images=n/a; annotations=n/a
- `data/valid/_annotations.coco.json`: skipped_large; images=n/a; annotations=n/a

## Stage 7 artifact excerpts (unverified report text)

- `README_STAGE7.md:108`: STAGE7_THREE_CANDIDATES_RECEIVED_PASS
- `README_STAGE7.md:109`: STAGE7_EVALUATION_GATE_PASS
- `README_STAGE7.md:110`: STAGE7_PRODUCTION_UNCHANGED_PASS
- `README_STAGE7.md:116`: Artifact: 'smart-parking-stage7-<run_id>-<attempt>'. Contains three candidate '.pth' files, train report, evaluation report, and MLflow audit summary. A Gate 'FAIL' for a
- `README_STAGE7.md:147`: 3. Kaggle processes the exact current-run private Dataset; all three candidates and SHA checks pass.
- `README_STAGE7_AUTO.md:52`: The GitHub workflow proceeds automatically to DVC pull, Kaggle training, Evaluation Gate, MLflow Tracking, **QA registration**. Registration is dynamic: MobileNetV3 / Res
- `README_STAGE7_AUTO.md:54`: If no candidate passes the gate, QA registration is skipped.
- `README_STAGE7_AUTO.md:65`: STAGE7_PRODUCTION_ALIAS_UNCHANGED
- `README_STAGE7_SIMULATION_UPDATE.md:8`: - 'pipeline/stage7_gate.py' evaluates baseline vs candidate and ranks eligible candidates by macro-F1 (tie break CPU p95). 'stage7_test_log.py' **preserves** suggested wi

## Git

- Status: modified

## What must still be verified manually

- Run local unit/integration tests and inspect their pass/fail reports.
- Start Streamlit and verify its buttons call the intended retraining workflow.
- Verify dataset labels, train/val/test split, data leakage, and DVC remote.
- Check MLflow/DagsHub live run and registry status.
- Check actual GitHub Actions run results, Stage 7 gate metrics and human approval.
- Confirm Production alias stays unchanged on FAIL.

## Safety and limitations

- File existence is not proof that a feature runs.
- No network/MLflow/DagsHub/GitHub checks were performed.
- No tests, retraining, inference, DVC actions or production promotion were run.
- No secret contents are printed.
