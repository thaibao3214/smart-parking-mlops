
import dagshub
import mlflow

dagshub.init(
    repo_owner="thaibao3214",
    repo_name="smart-parking-mlops",
    mlflow=True
)

print("MLflow Tracking URI:", mlflow.get_tracking_uri())
print("KET NOI DAGSHUB THANH CONG")
