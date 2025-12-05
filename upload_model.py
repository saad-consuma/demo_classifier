import os
from google.cloud import aiplatform

PROJECT_ID = "gemini-api-426311"
REGION = "us-central1"
IMAGE_URI = "us-central1-docker.pkg.dev/gemini-api-426311/region-classifier-repo/region-classifier-slm:v1"

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Get HF_TOKEN from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN env var not set")

    # Define environment variables for the container
    env_vars = {
        "HF_TOKEN": hf_token,
        "PORT": "8080",
        "MODEL_NAME": "meta-llama/Llama-3.2-1B-Instruct",
    }

    model = aiplatform.Model.upload(
        display_name="region-classifier-slm",
        serving_container_image_uri=IMAGE_URI,
        serving_container_ports=[8080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_environment_variables=env_vars,
    )

    print("Uploaded model resource name:")
    print(model.resource_name)

if __name__ == "__main__":
    main()
