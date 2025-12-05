from google.cloud import aiplatform

PROJECT_ID = "gemini-api-426311"
REGION = "us-central1"
ENDPOINT_ID = "REPLACE_WITH_ENDPOINT_ID"  # from deploy step

def main():
    client = aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    )

    endpoint_path = client.endpoint_path(
        project=PROJECT_ID,
        location=REGION,
        endpoint=ENDPOINT_ID,
    )

    # Example datapoint (same structure you tested locally)
    instance = {
        "id": "1862330364896698772",
        "metadata_content": {
            "content": "#shorts\nअगर कमजोरी मिटाना हो तो - ...",
            "created_at": "2024-11-29",
            "language": "hi",
            "type": "tweet"
        },
        "profile_stats": {
            "username": "RishiDarshan",
            "full_name": "Rishi Darshan™",
            "description": "Official account of Rishi Darshan...",
            "user_id": "1420127028",
            "followers": 36425,
            "following": 57,
            "location": "San Jose, CA",
            "is_verified": False,
        },
        "engagements": {
            "likes": 430,
            "comments": 49,
            "shares": 374,
            "quotes": 0,
            "bookmarks": 59,
            "views": 1211,
            "engagement_rate": 2.503774879890185,
        },
        "comments_collected": 36,
        "source": "twitter_1",
        "search_query": "indian gym supplements since:2024-10-22 lang:en until:2025-10-22 lang:en",
    }

    response = client.predict(
        endpoint=endpoint_path,
        instances=[instance],
    )

    print("Predictions:")
    print(response.predictions)

if __name__ == "__main__":
    main()
