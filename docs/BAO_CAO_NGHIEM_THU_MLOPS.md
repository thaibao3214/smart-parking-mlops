# Biên bản tổng kết thử nghiệm Smart Parking MLOps

**Phạm vi:** Đồ án tốt nghiệp sử dụng dữ liệu PKLot phát lại theo thời gian mô phỏng; **không** sử dụng camera thực tế. Đây là kiểm thử quy trình MLOps, không phải bằng chứng cải thiện trên dữ liệu ngoài miền PKLot.

## 1. Quy trình đã trình diễn

1. Streamlit mô phỏng luồng dữ liệu PKLot train theo chu kỳ và gọi FastAPI.
2. Đối chiếu nhãn với COCO Ground Truth, tạo dữ liệu huấn luyện lại.
3. Lưu phiên bản dữ liệu qua DVC, gửi GitHub Actions.
4. Kaggle chạy huấn luyện lại; MLflow lưu kết quả; Evaluation Gate so sánh các candidate.
5. Người dùng xem xét và chọn **dùng thử** candidate QA trên Streamlit.
6. FastAPI tải đúng phiên bản QA để suy luận, trong khi alias `production` được giữ nguyên.

## 2. Kết quả quan sát được

- Dữ liệu mô phỏng: **20 frame**, **2.000 crop** (theo trạng thái Streamlit).
- GitHub Actions Run #19: `45903240562`, trạng thái `Success`.
- Candidate đề xuất: **MobileNetV3**, phiên bản QA **6** trong ảnh kiểm tra Streamlit.
- Production khi kiểm thử: **MobileNetV3 phiên bản 1**.

| Mô hình | Macro F1 v1 | Macro F1 candidate | CPU p95 | Evaluation Gate |
|---|---:|---:|---:|---|
| MobileNetV3 | 1.0000 | 1.0000 | 4.35 ms | PASS |
| ResNet18 | 0.9922 | 0.9922 | 18.64 ms | PASS |
| Custom CNN | 0.9961 | 0.9922 | 8.40 ms | FAIL |

**Lưu ý:** Macro F1/Accuracy thuộc tập PKLot validation; không được hiểu là xác suất đúng của một ảnh bất kỳ. Việc hai mô hình PASS không chứng minh model mới tốt hơn v1; MobileNetV3 có F1 bằng v1 trong lần đánh giá này.

Nguồn tham chiếu: [GitHub Actions Run #19](https://github.com/thaibao3214/smart-parking-mlops/actions/runs/45903240562).

## 3. Phân biệt các trạng thái model

- **Production:** model thực sự đang phục vụ mặc định cho người dùng (v1 khi kiểm thử).
- **QA/dùng thử:** model mới đã qua Gate, được người dùng chốt trong **phiên Streamlit** (v6 khi kiểm thử).
- Chốt dùng thử **không đồng nghĩa** phê duyệt Production. Việc triển khai chính thức phải có quy trình phê duyệt, kiểm thử và rollback riêng.

## 4. Kiểm tra nghiệm thu sau cùng

Chạy script đi kèm khi FastAPI hoạt động:

```powershell
python scripts\final_acceptance_check.py
python scripts\final_acceptance_check.py --inference
```

Script kiểm tra DVC, Git, trạng thái Production/QA và (khi có `--inference`) dùng **cùng một ảnh PKLot test nguyên bản** để thử cả hai model. Không ghi dữ liệu hay thay model.

**Chỉ đánh dấu toàn bộ hệ thống PASS sau khi các phép kiểm tra trên máy cục bộ được thực hiện.**

## 5. Giới hạn khoa học

- Frame mô phỏng lấy từ PKLot train đã tồn tại từ trước. Chưa xác nhận toàn bộ ảnh batch tách khỏi dữ liệu v1 đã học.
- Chưa kiểm chứng trên camera thật, nên không suy rộng kết quả ra môi trường thực tế.
- Ground Truth dùng để chuẩn bị dữ liệu và đánh giá; FastAPI chỉ sử dụng ROI từ COCO để xác định ô đỗ, không đọc nhãn Empty/Occupied khi suy luận.
- Hai script Registry phụ chưa theo dõi bởi Git cần được xác minh mục đích trước khi quyết định lưu trữ; không tự chạy chúng.

## 6. Kịch bản trình diễn ngắn

**Camera mô phỏng → Huấn luyện & theo dõi → Đánh giá và chọn mô hình → Thử nhận diện → Kiểm tra phiên bản FastAPI trả về.**

Không cần chạy batch mới để bảo vệ đồ án. Có thể sử dụng kết quả đã lưu của lần huấn luyện thành công.
