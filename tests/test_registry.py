
import dagshub
import mlflow
import torch

# Ket noi DagsHub
dagshub.init(
    repo_owner="thaibao3214",
    repo_name="smart-parking-mlops",
    mlflow=True
)

# Tai model theo alias production
model_uri = "models:/SmartParking-MobileNetV3-Small@production"

print("Dang tai model tu DagsHub...")

model = mlflow.pytorch.load_model(
    model_uri,
    map_location="cpu"
)

# Kiem tra inference
x = torch.randn(1, 3, 64, 64)

with torch.no_grad():
    output = model(x)

print("Model:", model_uri)
print("Input:", x.shape)
print("Output:", output.shape)
print("Prediction:", output.argmax(dim=1).item())
print("REGISTRY INFERENCE SUCCESS")
