# Stage 7 — PKLot train camera simulation (TEST_ONLY)

This revision uses **all 2,000 manually confirmed crops from 20 PKLot TRAIN simulation frames**. It does not use previously collected PKLot TEST simulation crops.

- DVC: `simulation_train.dvc` and `confirmed_train_sim.dvc`, plus existing `data.dvc`.
- GitHub Actions packages authenticated sample provenance with original PKLot TRAIN frame checksums and the unchanged PKLot VALID technical holdout.
- Kaggle fine-tunes all three models and retains candidate checkpoints.
- `pipeline/stage7_gate.py` evaluates baseline vs candidate and ranks eligible candidates by macro-F1 (tie break CPU p95). `stage7_test_log.py` **preserves** suggested winner (previous revision accidentally nulled it).
- MLflow logs results as TEST_ONLY; **no candidate is registered or promoted automatically**. Model Registry versions can be added in a separately reviewed step.
- An existing real Stage 7 gate for external camera data remains untouched.
- PKLot valid was already used in prior experiments, hence this is NOT an unbiased claim of real-world generalization or improvement.

## Local commands

```powershell
cd C:\Users\thaib\Downloads\project
python -m py_compile pipeline/stage7_test_prepare.py kaggle/stage7_test_train.py pipeline/stage7_test_log.py
dvc add simulation_train
dvc add confirmed_train_sim
dvc push simulation_train.dvc confirmed_train_sim.dvc -r origin
git add simulation_train.dvc confirmed_train_sim.dvc pipeline/stage7_test_prepare.py kaggle/stage7_test_train.py pipeline/stage7_test_log.py .github/workflows/stage7_test_only.yml
git commit -m "Stage7 use confirmed PKLot train simulation TEST_ONLY"
git push github HEAD:main
```

Check your branch before the last push; `HEAD:main` assumes GitHub's intended branch is main. Do not `git add .`. Do not commit DagsHub tokens.

Run GitHub Actions -> `Smart Parking - Stage 7 TEST ONLY GPU Integration` -> `Run workflow`.
