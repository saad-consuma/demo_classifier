import os
from google.cloud import aiplatform

PROJECT_ID = "gemini-api-426311"
REGION = "us-central1"

# Model ID (no @1 needed)
MODEL_NAME = "projects/792829304616/locations/us-central1/models/1901406348633964544"

# Container image in Artifact Registry
IMAGE_URI = "us-central1-docker.pkg.dev/gemini-api-426311/region-classifier-repo/region-classifier-slm:v1"

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN env var not set")

    # Get model reference
    model = aiplatform.Model(model_name=MODEL_NAME)

    # Deploy with container override
    endpoint = model.deploy(
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=2,
        traffic_split={"0": 100},
        deploy_request_timeout=1800,   # IMPORTANT: model downloads can be slow
        # Override container
        container_spec={
            "image_uri": IMAGE_URI,
            "env": [
                {"name": "HF_TOKEN", "value": hf_token},
                {"name": "PORT", "value": "8080"},
                {"name": "MODEL_NAME", "value": "meta-llama/Llama-3.2-1B-Instruct"},
            ],
        },
    )

    print("Deployed endpoint resource name:")
    print(endpoint.resource_name)
    print("Endpoint ID:", endpoint.resource_name.split("/")[-1])


if __name__ == "__main__":
    main()
