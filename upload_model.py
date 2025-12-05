from google.cloud import aiplatform

PROJECT_ID = "gemini-api-426311"
REGION = "us-central1"
IMAGE_URI = "us-central1-docker.pkg.dev/gemini-api-426311/region-classifier-repo/region-classifier-slm:v1"

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.Model.upload(
        display_name="region-classifier-slm",
        serving_container_image_uri=IMAGE_URI,
        serving_container_ports=[8080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )

    print("Uploaded model resource name:")
    print(model.resource_name)

if __name__ == "__main__":
    main()
