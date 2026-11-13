# Smart Parking — Stage 6: Test evaluation + candidate review

## Mục tiêu

- Đánh giá sáu checkpoint (ba v1 và ba candidate Stage 3/4) trên **cùng 2.000 crop** (mặc định 1.000 Empty + 1.000 Occupied) từ `data/test` phiên bản DVC cố định.
- Loại hoàn toàn **20 ảnh gốc** đã sử dụng trong mô phỏng online, lấy từ `data/realtime_stream/manifest.json` (`image_files`). Nếu thiếu manifest, workflow dừng.
- Accuracy, macro Precision, macro Recall, macro F1, F1 từng lớp, confusion matrix và latency CPU (batch=1, 1 thread, warmup, đo forward riêng).
- So sánh candidate với v1 cùng kiến trúc: accuracy và macro F1 không giảm quá 0.005 (0.5 điểm phần trăm). Ngưỡng là lựa chọn kỹ thuật tạm thời, không phải chứng nhận thống kê.
- Đề xuất thứ tự **review** theo macro F1 rồi CPU p95; **không tự cập nhật MLflow Registry / Production**, kể cả khi gate PASS.
- Ghi metrics lên DagsHub MLflow, xuất JSON, CSV dự đoán từng ROI và GitHub Actions Summary.

## Cảnh báo phương pháp luận

Mặc dù Stage 3/4 không train bằng test, PKLot test đã được xem/truy vấn trong dự án trước đây, và các ROI cùng frame có tương quan. Do đó không gọi phép kiểm thử này là *blind external independent test*. Không đưa bất kỳ phần nào của test (gồm 2.000 online crops lấy từ test) vào retraining về sau. Trước promotion Production chính thức, nên có bộ kiểm thử mới từ camera/thời gian khác, tách ở cấp frame/camera/ngày và chưa dùng để điều chỉnh mô hình.

## Cài đặt

Giải nén vào thư mục gốc project để tạo:

```text
.github/workflows/stage6_independent_eval.yml
pipeline/stage6_evaluate.py
README_STAGE6.md
```

Yêu cầu các file đã có trên GitHub: `retraining/evaluate.py`, `data.dvc`, DVC pointers của ba checkpoint, `data/realtime_stream/manifest.json` nằm trong data do DVC quản lý; secrets `DAGSHUB_TOKEN`. `gh run download` lấy artifact cố định của các run Stage 3 (`43458044027`) và Stage 4 (`43479352874`). Các artifact cần còn hiệu lực.

```powershell
cd C:\Users\thaib\Downloads\project
python -m py_compile pipeline/stage6_evaluate.py
git add .github/workflows/stage6_independent_eval.yml
git add pipeline/stage6_evaluate.py
git add README_STAGE6.md
git diff --cached --name-only
git commit -m "feat: Stage 6 test evaluation and model review"
git push github main
```

Trên GitHub Actions, chạy **Smart Parking - Stage 6 Test Evaluation and Model Review** / `workflow_dispatch` / `main`, mặc định `per_class=1000`. Nếu ít hơn 1.000 bbox/lớp khả dụng sau khi loại 20 frame, workflow dừng thay vì thay đổi mẫu ngầm; giảm per_class một cách minh bạch rồi chạy lại.

## Kết quả

- `stage6_report.json`: bảng số liệu đầy đủ, confusion matrix, latency v1/candidate, gate, recommendation.
- `stage6_predictions.csv`: mỗi ROI test và dự đoán của cả sáu checkpoint.
- `stage6_summary.md`: bảng cho GitHub Summary.
- MLflow experiment: `smart-parking-stage6-independent-evaluation`.

### Marker thành công

```text
STAGE6_CPU_DEPS_PASS
STAGE6_DVC_PULL_PASS
STAGE6_CANDIDATES_DOWNLOADED_PASS
STAGE6_SIX_CHECKPOINT_HASHES_PASS
STAGE6_ONLINE_SOURCE_EXCLUSION_PASS
STAGE6_EVAL_SELECTION_PASS
STAGE6_METRICS_AND_CPU_LATENCY_PASS
STAGE6_PRODUCTION_UNCHANGED_PASS
STAGE6_MLFLOW_TRACKING_PASS
STAGE6_END_TO_END_PASS
```

Thận trọng: **SUCCESS** nghĩa đánh giá hoàn tất; không chứng minh mô hình mới vượt baseline hoặc đã Production-ready.
