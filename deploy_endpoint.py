import os
from google.cloud import aiplatform

PROJECT_ID = "gemini-api-426311"
REGION = "us-central1"

# Model ID (no @1 needed)
MODEL_NAME = ""

# Container image in Artifact Registry
IMAGE_URI = "us-central1-docker.pkg.dev/gemini-api-426311/region-classifier-repo/region-classifier-slm:v1"

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN env var not set")

    # Get model reference
    model = aiplatform.Model(model_name=MODEL_NAME)

    # Deploy the model
    # NOTE: Environment variables must be set during model upload, not deployment
    # If you need to pass env vars, you must re-upload the model with serving_container_environment_variables
    endpoint = model.deploy(
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=2,
        traffic_split={"0": 100},
        deploy_request_timeout=1800,   # IMPORTANT: model downloads can be slow
    )

    print("Deployed endpoint resource name:")
    print(endpoint.resource_name)
    print("Endpoint ID:", endpoint.resource_name.split("/")[-1])


if __name__ == "__main__":
    main()
