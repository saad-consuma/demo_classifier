# call_endpoint.py
import argparse
import json
from google.cloud import aiplatform

PROJECT_ID = "gemini-api-426311"
REGION = "us-central1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-id", required=True)
    args = parser.parse_args()

    client = aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    )

    endpoint_path = client.endpoint_path(
        project=PROJECT_ID,
        location=REGION,
        endpoint=args.endpoint_id,
    )

    instances = [
        {
            "id": "test",
            "metadata_content": {"content": "What do you think about fitness in Indian youths?"},
            "profile_stats": {"username": "test_user"},
            "engagements": {},
            "source": "local",
        },
        {
            "id": "test2",
            "metadata_content": {"content": "why is the weather in Scarborough fucked? why cant it be like missasauga?"},
            "profile_stats": {"username": "test_user"},
            "engagements": {},
            "source": "local",
        }
    ]

    resp = client.predict(endpoint=endpoint_path, instances=[json.dumps(instance) for instance in instances])
    print(resp.predictions)

if __name__ == "__main__":
    main()
