STAGE 3 - FINE-TUNE MOBILE NET V3 v1 ON KAGGLE GPU

IMPORTANT: The existing Kaggle Stage 2 workflow and production Registry are NOT modified.
No PKLot test frames or the existing 2,000 simulated-online test-derived crops are used.
This smoke fine-tunes the ORIGINAL v1 checkpoint on 128 PKLot train crops and validates
on 64 PKLot validation crops. It is NOT the complete online retraining phase.

ONE-TIME LOCAL SETUP in PowerShell, inside C:\Users\thaib\Downloads\project:

  Test-Path mobilenet_model/mobilenet_v3_small_best.pth
  # Must return True; the file is the original training checkpoint (not MLflow PT2 export).
  dvc add mobilenet_model/mobilenet_v3_small_best.pth
  dvc push mobilenet_model/mobilenet_v3_small_best.pth.dvc -r origin

Copy the 2 new files from this ZIP into the same relative paths:
  .github/workflows/stage3_finetune.yml
  kaggle/stage3_finetune.py

Push only pointers, scripts, and workflow to GitHub (never the .pth):
  git add mobilenet_model/mobilenet_v3_small_best.pth.dvc .gitignore
  git add .github/workflows/stage3_finetune.yml kaggle/stage3_finetune.py
  git status --short
  git commit -m "feat: Stage 3 original v1 checkpoint fine-tuning"
  git push github main

If git add .gitignore reports an unmatched path, omit that path; DVC can put an
ignore file in the checkpoint directory instead. In that case add that generated
mobilenet_model/.gitignore file instead. Check git status and ensure no .pth is staged.

On GitHub Actions launch:
  Smart Parking - Stage 3 v1 Checkpoint Fine-Tune

Expected key markers:
  STAGE3_V1_CHECKPOINT_DVC_PULL_PASS
  STAGE3_CHECKPOINT_SHA256_PASS
  STAGE3_V1_CHECKPOINT_LOAD_PASS
  STAGE3_FINETUNE_PASS
  STAGE3_FINETUNE_END_TO_END_PASS

The output is stage3_candidate_mobilenetv3.pth + stage3_report.json in the
GitHub Actions diagnostic artifact. No Production registry update is performed.

NOTE: DVC credentials already in GitHub Secret DAGSHUB_TOKEN and Kaggle API Token
in GitHub Secret KAGGLE_API_TOKEN are reused. Do not commit credentials.
