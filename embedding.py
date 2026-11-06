import json
import boto3
from typing import List
from langchain_core.embeddings import Embeddings


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

            result = json.loads(response["body"].read())

            embeddings.append(result["embedding"])

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


# Initialize
embeddings_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="ap-south-1"
)


# Test
texts = [
    "AWS EC2 provides virtual servers.",
    "Amazon S3 stores objects in the cloud.",
    "Machine learning uses data to train models."
]

vectors = embeddings_model.embed_documents(texts)

print("Number of embeddings:", len(vectors))
print("Embedding dimensions:", len(vectors[0]))
print("First 5 values:", vectors[0][:5])

query_vector = embeddings_model.embed_query("What is EC2?")

print("Query dimensions:", len(query_vector))

print("\nSUCCESS: Bedrock embeddings generated!")
