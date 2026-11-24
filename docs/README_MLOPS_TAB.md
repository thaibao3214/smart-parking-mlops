# Smart Parking – tích hợp tab MLOps Monitoring

Thay `dashboard/app.py` bằng `app.py` trong ZIP, thêm `dashboard/mlops_monitoring.py`.
Không cần thay `dashboard/online_stream.py` (đính kèm nguyên bản để đối chiếu).

## Thiết lập

```powershell
cd C:\Users\thaib\Downloads\project
python -m py_compile dashboard\app.py dashboard\mlops_monitoring.py dashboard\online_stream.py
python -m pip install mlflow
streamlit run dashboard\app.py
```

MLflow cần quyền truy cập DagsHub. Nếu môi trường chưa có sẵn xác thực, thiết lập biến môi trường trong phiên PowerShell chạy Streamlit:

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/thaibao3214/smart-parking-mlops.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<DagsHub username>"
$env:MLFLOW_TRACKING_PASSWORD="<DagsHub token>"
```

Không nhập hoặc in token trong chat, không lưu token vào Git. Nếu MLflow chưa kết nối được, tab vẫn hiển thị Batch marker nội bộ và GitHub Actions công khai.

Tab mới chỉ đọc dữ liệu, không tự retrain, promote hoặc xoá file. Metric lấy từ MLflow Run, không dùng số liệu hardcode. GitHub Actions đọc từ API công khai và có thể bị giới hạn truy cập. Batch history đọc từ `pipeline/stage7_batches/*.json`.

**Chú ý:** Đây là module giám sát, không phải kiểm thử đầy đủ với môi trường Windows/FastAPI/DagsHub thực tế của bạn.
