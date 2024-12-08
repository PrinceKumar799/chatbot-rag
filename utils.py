from mistralai import Mistral
import requests
import numpy as np
import faiss
import os

def get_text_embedding(input,client):
    try:
        embeddings_batch_response = client.embeddings.create(
            model="mistral-embed",
            inputs=input
        )
        return embeddings_batch_response.data[0].embedding
    except Exception as e:
        print(f"Mistral request exception: {e}",)

