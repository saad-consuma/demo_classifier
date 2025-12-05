import os
from google.cloud import aiplatform

PROJECT_ID = "gemini-api-426311"
REGION = "us-central1"

MODEL_ID = "projects/792829304616/locations/us-central1/models/1901406348633964544"  # just the numeric ID, e.g. 1234567890123456789

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.Model(model_name=MODEL_ID)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN env var not set")

    endpoint = model.deploy(
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=3,
        traffic_split={"0": 100},
        env={
            "HF_TOKEN": hf_token,
            "PORT": "8080",
            "MODEL_NAME": "meta-llama/Llama-3.2-1B-Instruct",
        },
    )

    print("Deployed endpoint resource name:")
    print(endpoint.resource_name)
    print("Endpoint ID:", endpoint.resource_name.split("/")[-1])

if __name__ == "__main__":
    main()
