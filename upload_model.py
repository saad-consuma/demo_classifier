# upload_model.py
import argparse
import os
from google.cloud import aiplatform

PROJECT_ID = "gemini-api-426311"
REGION = "us-central1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", required=True, help="Container image URI in Artifact Registry")
    parser.add_argument("--display-name", default="region-classifier-slm", help="Vertex model display name")
    args = parser.parse_args()

    aiplatform.init(project=PROJECT_ID, location=REGION)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN env var not set")

    model = aiplatform.Model.upload(
        display_name=args.display_name,
        serving_container_image_uri=args.image_uri,
        serving_container_ports=[8080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_environment_variables={
            "HF_TOKEN": hf_token,
            "PORT": "8080",
            "MODEL_NAME": "meta-llama/Llama-3.2-3B-Instruct",
            "MAX_INTERNAL_BATCH": "8",
        },
    )

    print("Uploaded model resource name:")
    print(model.resource_name)

if __name__ == "__main__":
    main()
 