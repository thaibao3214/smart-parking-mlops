# Báo cáo kiểm thử Stage 7 — Retraining TEST ONLY

**Ngày:** 24/11/2026
**Hệ thống:** Smart Parking MLOps
**Dữ liệu:** PKLot train mô phỏng camera; batch_004 (19 frames hợp lệ, 1.900 crops, 1 frame loại trừ)
**Chế độ:** `TEST_ONLY_NO_PROMOTION`

## Mục tiêu

Kiểm chứng việc kích hoạt huấn luyện lại từ Streamlit thông qua GitHub Actions và Kaggle GPU; đánh giá các candidate trước khi cho phép đưa vào Production.

## Kết quả kiểm thử

Theo ảnh GitHub Actions người thực hiện cung cấp, Run #17 của workflow Stage 7 TEST ONLY GPU Integration hiển thị **Success** (10 phút 57 giây). Streamlit trước đó ghi nhận yêu cầu **`dispatched`**. Việc liên kết chính xác Run #17 với batch_004 cần đối chiếu marker `STAGE7_RESOLVED_IMMUTABLE_BATCH` trong job logs.

| Model | v1 macro F1 | Candidate macro F1 | CPU p95 | Evaluation Gate |
|---|---:|---:|---:|---|
| MobileNetV3 | 1.0000 | 0.9961 | 3.82 ms | FAIL |
| ResNet18 | 0.9922 | 0.9844 | 16.83 ms | FAIL |
| Custom CNN | 0.9961 | 0.9922 | 7.07 ms | FAIL |

**Candidate đề xuất review:** Không có.
**Production:** Không thay đổi alias `production` theo báo cáo run được cung cấp.
**Đánh giá:** Validation sử dụng PKLot validation crops. Final deployment cần phê duyệt thủ công và workflow triển khai riêng.

## Diễn giải

- **Workflow Success:** pipeline huấn luyện/đánh giá và logging chạy xong, không có nghĩa candidate tốt hơn v1.
- **Gate FAIL:** cả ba candidate có Macro F1 thấp hơn baseline; do đó không chọn ứng viên để promotion.
- **TEST ONLY:** dùng dữ liệu train PKLot để mô phỏng camera, không đại diện cho dữ liệu camera mới ngoài thực tế.
- Việc dùng lại dữ liệu train chỉ thích hợp để kiểm tra luồng hệ thống; cần giữ validation/test độc lập và kiểm soát rò rỉ dữ liệu khi đánh giá chất lượng mô hình.

## Kết luận và việc còn lại

Pipeline Retraining TEST ONLY đã có bằng chứng hoạt động qua Streamlit/GitHub Actions/Kaggle/Evaluation Gate. Hoàn thiện Monitoring bằng cách đọc trạng thái run và artifact kết quả từ GitHub, đối chiếu đúng batch ID; đồng thời kiểm tra chức năng này trực tiếp trên máy triển khai. Không tự động thay đổi Production.
