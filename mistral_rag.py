from mistralai import Mistral
import requests
import numpy as np
import faiss
import os
from dotenv import load_dotenv
from utils import get_text_embedding

load_dotenv()

def creat_chunks():
    with open('doc.txt', 'r') as file:
        text = file.read()
    chunk_size = 2048
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks
    

def load_embeddings_db(text_embeddings):
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)
    faiss.write_index(index, "faiss_index.bin")
    
if __name__ == "__main__":
    client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
    chunks = creat_chunks()
    print("chunks: ",len(chunks))
    text_embeddings = np.array([get_text_embedding(chunk,client) for chunk in chunks])
    print("embeddings: ",text_embeddings)
    load_embeddings_db(text_embeddings)

    