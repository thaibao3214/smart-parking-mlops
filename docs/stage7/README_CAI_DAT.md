# Stage 7 — Monitoring trên Streamlit

## Mục tiêu

Theo dõi run `workflow_dispatch` của Stage 7 TEST ONLY mà không chạy lại retraining. Chỉ liên kết run với batch khi nhật ký GitHub chứa marker `STAGE7_RESOLVED_IMMUTABLE_BATCH` đúng `batch_name` và `batch_id`.

## Cài đặt (PowerShell trong thư mục project)

1. Dừng Streamlit bằng Ctrl+C.
2. Sao lưu code hiện có:

```powershell
Copy-Item dashboard\stage7_control.py dashboard\stage7_control.py.bak -Force
Copy-Item dashboard\stage7_control_worker.py dashboard\stage7_control_worker.py.bak -Force
```

3. Giải nén gói và copy **3 file `.py`** vào thư mục `dashboard\` (ghi đè 2 file cùng tên, thêm mới `stage7_monitor.py`). Không chép README vào `dashboard` nếu không cần.
4. Kiểm tra cú pháp:

```powershell
python -m py_compile dashboard\stage7_control.py dashboard\stage7_control_worker.py dashboard\stage7_monitor.py
```

5. Mở PowerShell **mới** để nhận GH_TOKEN đã cấu hình ở cấp User. Nếu cần, đặt lại token cho phiên bằng cách nhập ẩn (không hiển thị trên màn hình):

```powershell
$env:GH_TOKEN = Read-Host 'GitHub token' -MaskInput
$env:MLOPS_CONTROL_PASSWORD = Read-Host 'MLOps password' -MaskInput
python -m streamlit run dashboard\app.py
```

Nếu dùng Windows PowerShell 5.1, `Read-Host -MaskInput` có thể không hỗ trợ; dùng `Read-Host -AsSecureString` và xử lý riêng, không in token ra màn hình. Với token lưu tại User environment variable, mở terminal mới và **không cần nhập lại** token. Mật khẩu MLOps vẫn cần cấu hình theo cách của bạn.

## Thao tác

- MLOps Control Center → Continuous Training → nhập mật khẩu.
- Cuộn đến **Giám sát Retraining trên GitHub Actions**.
- Chọn bản ghi `batch_004 · dispatched` tương ứng.
- **Kiểm tra run trên GitHub**: đối chiếu marker trong job log; nếu không tìm thấy thì KHÔNG tự gán Run #17.
- Khi run đã `completed`, chọn **Lấy kết quả Evaluation Gate**. Việc này tải artifact của run (có cả 3 model `.pth`, có thể lớn).
- Xem Macro F1/Gate/Production ngay dưới nút; có thể tải báo cáo `.md`.

Không cần nhấn nút đỏ **Chạy retraining** khi chỉ muốn xem kết quả cũ.

## Lưu ý

- Monitor chỉ đọc API GitHub; không thay đổi alias `production`.
- Cần `GH_TOKEN` còn hiệu lực và có quyền Actions: read để đọc logs/artifacts. API có thể bị rate-limit hoặc tạm lỗi.
- Artifact có `retention-days: 14` trong workflow đã cung cấp; sau khi hết hạn, không lấy được kết quả cũ bằng cách này.
- Nếu artifact vượt 90 MB, monitor cố ý không tải để tránh tốn RAM. Xem Job Summary trên GitHub thay thế.
- `dispatched` = GitHub nhận yêu cầu; `completed/success` = workflow thành công; `Gate PASS/FAIL` = đánh giá chất lượng model, là hai khái niệm khác nhau.
