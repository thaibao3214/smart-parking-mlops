import json
import os
import io
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, resnet18
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI(title="Smart Parking MLOps PKLot API", version="3.0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "data/test"
JSON_PATH = os.path.join(TEST_DIR, "_annotations.coco.json")

PATH_MAP = {
    "mobilenetv3": "mobilenet_model/mobilenet_v3_small_best.pth",
    "resnet18": "resnet_model/resnet18_best.pth",
    "custom_cnn": "cnn_model/optimized_custom_cnn_best.pth",
}

# --- ĐỊNH NGHĨA CUSTOM CNN ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
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

def load_trained_model(model_type, weight_path):
    if model_type == "mobilenetv3":
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    elif model_type == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_type == "custom_cnn":
        model = OptimizedPKLotCNN(num_classes=2)
    else:
        raise ValueError(f"Model type không hợp lệ: {model_type}")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số tại: {weight_path}")

    checkpoint = torch.load(weight_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
def home():
    return {"message": "Smart Parking MLOps PKLot API is running!"}

@app.post("/predict")
async def predict_parking(
    model_type: str = Form(...), 
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    image_name = file.filename
    
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((640, 640))
    
    bboxes = []
    
    # Tự động tra cứu tọa độ ô đỗ dựa trên tên file ảnh trong COCO JSON của tập PKLot
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
        target_image_id = None
        for img_info in coco_data["images"]:
            if img_info["file_name"] == image_name:
                target_image_id = img_info["id"]
                break
        if target_image_id is not None:
            bboxes = [
                ann["bbox"] for ann in coco_data.get("annotations", [])
                if ann["image_id"] == target_image_id
            ]

    if not bboxes:
        return {"error": f"Không tìm thấy thông tin tọa độ ô đỗ cho ảnh '{image_name}' trong tập dữ liệu chuẩn PKLot (COCO JSON)."}

    # Load mô hình AI
    weight_path = PATH_MAP.get(model_type, PATH_MAP["mobilenetv3"])
    try:
        model = load_trained_model(model_type, weight_path)
    except Exception as e:
        return {"error": str(e)}
        
    count_empty = 0
    count_occupied = 0
    predictions = []
    
    with torch.no_grad():
        for idx, bbox in enumerate(bboxes):
            x, y, w, h = map(int, bbox)
            xmin, ymin = max(0, x), max(0, y)
            xmax, ymax = min(640, x + w), min(640, y + h)

            if (xmax - xmin) <= 5 or (ymax - ymin) <= 5:
                continue

            crop = image_pil.crop((xmin, ymin, xmax, ymax))
            crop_tensor = transform(crop).unsqueeze(0).to(DEVICE)

            outputs = model(crop_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()

            if pred_label == 1:
                status = "Occupied"
                count_occupied += 1
            else:
                status = "Empty"
                count_empty += 1

            predictions.append({
                "id": idx + 1,
                "status": status,
                "box": [xmin, ymin, xmax - xmin, ymax - ymin]
            })
            
    total_slots = count_empty + count_occupied
    occupancy_rate = (count_occupied / total_slots) * 100 if total_slots > 0 else 0.0

    return {
        "model_used": model_type,
        "image_name": image_name,
        "total_spaces": total_slots,
        "occupied_spaces": count_occupied,
        "empty_spaces": count_empty,
        "occupancy_rate": round(occupancy_rate, 2),
        "predictions": predictions
    }