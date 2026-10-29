import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ==========================================
# 1. BƯỚC VẼ TỌA ĐỘ Ô ĐỖ XE (ĐÃ RESIZE MÀN HÌNH)
# ==========================================
points = []
slots_original = [] # Lưu tọa độ trên ảnh gốc

def click_event(event, x, y, flags, params):
    """Bắt sự kiện click chuột trên ảnh đã resize và quy đổi về ảnh gốc."""
    global points, slots_original, img_display, scale_x, scale_y
    if event == cv2.EVENT_LBUTTONDOWN:
        # Lưu điểm hiển thị để vẽ
        points.append((x, y))
        cv2.circle(img_display, (x, y), 4, (0, 255, 0), -1)
        
        # Đủ 4 góc -> Quy đổi tọa độ về kích thước ảnh GỐC
        if len(points) == 4:
            xs_disp = [p[0] for p in points]
            ys_disp = [p[1] for p in points]
            
            # Tính min/max trên màn hình hiển thị
            xmin_d, xmax_d = min(xs_disp), max(xs_disp)
            ymin_d, ymax_d = min(ys_disp), max(ys_disp)
            
            # Quy đổi ngược về tọa độ gốc
            xmin_orig = int(xmin_d * scale_x)
            xmax_orig = int(xmax_d * scale_x)
            ymin_orig = int(ymin_d * scale_y)
            ymax_orig = int(ymax_d * scale_y)
            
            slots_original.append((xmin_orig, ymin_orig, xmax_orig, ymax_orig))
            
            # Vẽ khung lên màn hình hiển thị
            cv2.rectangle(img_display, (xmin_d, ymin_d), (xmax_d, ymax_d), (255, 0, 0), 2)
            cv2.putText(img_display, f"Slot {len(slots_original)}", (xmin_d, ymin_d - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            points = [] # Reset cho ô tiếp theo
            
        cv2.imshow("Danh dau o do xe (Nhan 'q' de xong)", img_display)

def mark_slots(image_path, max_width=1280, max_height=720):
    """Tự động resize ảnh vừa màn hình để chấm chuột dễ dàng."""
    global img_display, scale_x, scale_y
    
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")
        
    h_orig, w_orig = orig_img.shape[:2]
    
    # Tính tỷ lệ thu nhỏ để vừa màn hình (mặc định tối đa 1280x720)
    scale_w = max_width / w_orig
    scale_h = max_height / h_orig
    scale = min(scale_w, scale_h, 1.0) # Không phóng to nếu ảnh nhỏ hơn 720p
    
    display_w = int(w_orig * scale)
    display_h = int(h_orig * scale)
    
    # Hệ số quy đổi tọa độ màn hình -> ảnh gốc
    scale_x = w_orig / display_w
    scale_y = h_orig / display_h
    
    # Resize ảnh để hiển thị vừa vặn
    img_display = cv2.resize(orig_img, (display_w, display_h))
    
    cv2.imshow("Danh dau o do xe (Nhan 'q' de xong)", img_display)
    cv2.setMouseCallback("Danh dau o do xe (Nhan 'q' de xong)", click_event)
    
    print(f"📐 Ảnh gốc ({w_orig}x{h_orig}) -> Đã thu nhỏ vừa màn hình ({display_w}x{display_h}).")
    print("👉 HƯỚNG DẪN: Click chuột trái 4 điểm xung quanh từng ô đỗ xe.")
    print("👉 Bấm phím 'q' trên bàn phím khi đánh dấu xong để chạy AI.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()
    return orig_img, slots_original

# ==========================================
# 2. CHUẨN BỊ MÔ HÌNH VÀ TRANSFORM
# ==========================================
def load_trained_model(model_path, device):
    from torchvision.models import mobilenet_v3_small
    
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. MAIN INFERENCE
# ==========================================
if __name__ == "__main__":
    CUSTOM_IMAGE_PATH = r"C:\Users\thaib\Downloads\project\Feature-How-much-does-it-cost-to-build-a-car-park.jpg"    # Ảnh tải trên mạng
    MODEL_WEIGHTS_PATH = r"C:\Users\thaib\Downloads\project\mobilenet_model\mobilenet_v3_small_best.pth" # File weights
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Bước 1: Đánh dấu ô đỗ xe (Ảnh to sẽ tự động thu nhỏ vừa màn hình)
    original_img, slot_boxes = mark_slots(CUSTOM_IMAGE_PATH, max_width=1280, max_height=720)
    
    if len(slot_boxes) == 0:
        print("❌ Chưa chọn ô đỗ xe nào!")
        exit()
        
    # Bước 2: Load mô hình PyTorch
    print(f"\n🔄 Loading model from {MODEL_WEIGHTS_PATH}...")
    model = load_trained_model(MODEL_WEIGHTS_PATH, DEVICE)
    
    # Bước 3: Crop ảnh gốc & Predict
    print(f"🚀 Bắt đầu dự đoán {len(slot_boxes)} ô đỗ...")
    output_img = original_img.copy()
    occupied_count, empty_count = 0, 0
    
    for idx, (xmin, ymin, xmax, ymax) in enumerate(slot_boxes):
        # Crop chính xác trên ảnh gốc
        crop = original_img[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue
            
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            status = predicted.item()
            
        # Đỏ (Có xe - 1), Xanh (Trống - 0)
        if status == 1:
            color, label = (0, 0, 255), "Occupied"
            occupied_count += 1
        else:
            color, label = (0, 255, 0), "Empty"
            empty_count += 1
            
        # Vẽ độ dày nét phù hợp với độ phân giải ảnh gốc
        thickness = max(2, int(original_img.shape[1] / 1000))
        font_scale = max(0.5, original_img.shape[1] / 2000)
        
        cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.putText(output_img, f"#{idx+1}: {label}", (xmin, max(ymin - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    # Bước 4: Thống kê & Lưu ảnh
    print("\n================ THỐNG KÊ ================")
    print(f"Tổng ô: {len(slot_boxes)} | 🔴 Có xe: {occupied_count} | 🟢 Trống: {empty_count}")
    print(f"📊 Tỷ lệ lấp đầy: {occupied_count/len(slot_boxes)*100:.1f}%")
    print("==========================================")
    
    cv2.imwrite("result_custom_test.jpg", output_img)
    print("✅ Đã lưu kết quả vào 'result_custom_test.jpg'")
    
    # Hiển thị kết quả (cũng resize để vừa màn hình)
    res_h, res_w = output_img.shape[:2]
    scale_res = min(1280/res_w, 720/res_h, 1.0)
    res_display = cv2.resize(output_img, (int(res_w*scale_res), int(res_h*scale_res)))
    
    cv2.imshow("Ket qua Nhan dien AI", res_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()