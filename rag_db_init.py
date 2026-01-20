import json
import chromadb
from sentence_transformers import SentenceTransformer

with open("sign_knowledge.json", "r") as f:
    sign_data = json.load(f)

client = chromadb.PersistentClient(
   path="rag_db"
)

collection = client.get_or_create_collection("signhands")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
ids = []

for sign, info in sign_data.items():
    text = f"""
    Sign: {sign}
    Meaning: {info['meaning']}
    Context: {info['context']}
    Sentences: {' '.join(info['sentences'])}
    """
    documents.append(text)
    ids.append(sign)

embeddings = embedder.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids
)

#client.persist()
#print("✅ RAG DB created for SignHands labels")
