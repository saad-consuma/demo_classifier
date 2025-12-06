# deploy_endpoint.py
import argparse
from google.cloud import aiplatform

PROJECT_ID = "gemini-api-426311"
REGION = "us-central1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Full Vertex model resource name")
    args = parser.parse_args()

    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.Model(model_name=args.model_name)

    endpoint = model.deploy(
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=2,
        traffic_split={"0": 100},
        deploy_request_timeout=1800,
    )

    print("Deployed endpoint resource name:")
    print(endpoint.resource_name)
    print("Endpoint ID:", endpoint.resource_name.split("/")[-1])

if __name__ == "__main__":
    main()
