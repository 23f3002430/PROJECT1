
import json
import boto3
from typing import List
from langchain_core.embeddings import Embeddings
from botocore.exceptions import ClientError


class BedrockEmbeddings(Embeddings):

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region_name: str = "ap-south-1"
    ):
        self.model_id = model_id

        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:

        embeddings = []

        # Titan accepts one input text per invocation
        for text in texts:

            body = json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                accept="application/json",
                contentType="application/json"
            )

            response_body = json.loads(
                response["body"].read()
            )

            embeddings.append(
                response_body["embedding"]
            )

        return embeddings

    def embed_documents(
        self, texts: List[str]
    ) -> List[List[float]]:

        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:

        return self._embed([text])[0]


# Initialize embedding model
embeddings_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="ap-south-1"
)



from sklearn.metrics.pairwise import cosine_similarity

# Test sentences
texts = [
    "AWS EC2 is a cloud computing service.",
    "Amazon EC2 provides virtual servers in the cloud.",
    "I like eating pizza."
]

# Generate embeddings
vectors = embeddings_model.embed_documents(texts)

# Check dimensions
print("Number of sentences:", len(vectors))
print("Embedding dimensions:", len(vectors[0]))

# Print first 5 values
print("\nFirst 5 embedding values:")
print(vectors[0][:5])

# Test query embedding
query = "What is Amazon EC2?"
query_vector = embeddings_model.embed_query(query)

print("\nQuery embedding dimensions:", len(query_vector))

# Calculate semantic similarity
similarities = cosine_similarity(
    [query_vector],
    vectors
)[0]

print("\nSimilarity scores:")

for text, score in zip(texts, similarities):
    print(f"{score:.4f} -> {text}")
