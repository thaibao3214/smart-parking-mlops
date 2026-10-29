import json
import os
import random
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import mobilenet_v3_small, resnet18

# 1. THIẾT LẬP CẤU HÌNH & ĐƯỜNG DẪN
TEST_DIR = "data/test"
JSON_PATH = os.path.join(TEST_DIR, "_annotations.coco.json")

# Chọn 1 trong 3 mô hình: 'mobilenetv3', 'resnet18', 'custom_cnn'
MODEL_TYPE = "mobilenetv3"

PATH_MAP = {
    "mobilenetv3": "mobilenet_model/mobilenet_v3_small_best.pth",
    "resnet18": "resnet_model/resnet18_best.pth",
    "custom_cnn": "cnn_model/optimized_custom_cnn_best.pth",
}

MODEL_PATH = PATH_MAP[MODEL_TYPE]
OUTPUT_PATH = "result_random_test.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- ĐỊNH NGHĨA LẠI CUSTOM CNN CHO HÀM LOAD ---
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.conv(x)


class OptimizedPKLotCNN(nn.Module):

    def __init__(self, num_classes=2):
        super(OptimizedPKLotCNN, self).__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# 2. HÀM LOAD MÔ HÌNH DÙNG CHUNG
def load_trained_model(model_type, weight_path):
    if model_type == "mobilenetv3":
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, 2
        )
    elif model_type == "resnet18":
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif model_type == "custom_cnn":
        model = OptimizedPKLotCNN(num_classes=2)

    checkpoint = torch.load(weight_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


transform = T.Compose(
    [
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 3. ĐỌC JSON VÀ LẤY ẢNH NGẪU NHIÊN
with open(JSON_PATH, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

random_image_info = random.choice(coco_data["images"])
image_name = random_image_info["file_name"]
target_image_id = random_image_info["id"]

image_path = os.path.join(TEST_DIR, image_name)
print(f"-> Đang test mô hình: {MODEL_TYPE.upper()}")
print(f"-> Chọn ngẫu nhiên ảnh: {image_name}")

# Lấy thông tin bounding box
bboxes = [
    ann["bbox"]
    for ann in coco_data.get("annotations", [])
    if ann["image_id"] == target_image_id
]
print(f"-> Số lượng ô đỗ detected: {len(bboxes)}")

# 4. CHẠY SUY LUẬN (INFERENCE)
model = load_trained_model(MODEL_TYPE, MODEL_PATH)

# Ép ảnh về chuẩn 640x640 khớp với định dạng dataset gốc
image_pil = Image.open(image_path).convert("RGB").resize((640, 640))
image_cv = cv2.imread(image_path)
image_cv = cv2.resize(image_cv, (640, 640))

count_empty, count_occupied = 0, 0

with torch.no_grad():
    for bbox in bboxes:
        x, y, w, h = map(int, bbox)

        # Giới hạn tọa độ trong khoảng khung hình 640x640
        xmin, ymin = max(0, x), max(0, y)
        xmax, ymax = min(640, x + w), min(640, y + h)

        if (xmax - xmin) <= 5 or (ymax - ymin) <= 5:
            continue

        crop = image_pil.crop((xmin, ymin, xmax, ymax))
        crop_tensor = transform(crop).unsqueeze(0).to(DEVICE)

        outputs = model(crop_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()

        # Class 1 = Occupied (Có xe - Đỏ), Class 0 = Empty (Trống - Xanh)
        if pred_label == 1:
            color = (0, 0, 255)  # Đỏ
            count_occupied += 1
        else:
            color = (0, 255, 0)  # Xanh lá
            count_empty += 1

        cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), color, 2)

# 5. XUẤT KẾT QUẢ
total_slots = count_empty + count_occupied
print(f"\n--- KẾT QUẢ DỰ ĐOÁN ({MODEL_TYPE.upper()}) ---")
print(f"Ô trống (Xanh lá): {count_empty}")
print(f"Ô có xe (Đỏ): {count_occupied}")
if total_slots > 0:
    print(f"Tỷ lệ lấp đầy: {(count_occupied / total_slots) * 100:.2f}%")

cv2.imwrite(OUTPUT_PATH, image_cv)
print(f"-> Đã lưu ảnh kết quả vào tệp: {OUTPUT_PATH}")