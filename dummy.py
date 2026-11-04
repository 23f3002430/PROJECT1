
import boto3
import os

# AWS Bedrock configuration
REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")

client = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION
)

# Replace with a model ID available in your Bedrock region
MODEL_ID = "google.gemma-3-4b-it"

def generate_response(prompt):

    try:
        response = client.converse(
            modelId=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": prompt}
                    ]
                }
            ],
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.7
            }
        )

        result = response["output"]["message"]["content"][0]["text"]

        return result

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":

    print("AWS Bedrock Connected!")
    print("Type 'exit' to stop.")

    while True:

        prompt = input("\nYou: ")

        if prompt.lower() == "exit":
            break

        answer = generate_response(prompt)

        print("\nGemma:", answer)
