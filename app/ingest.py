from app.rag import ingest_directory

if __name__ == "__main__":
    n = ingest_directory("data/sample_docs")
    print(f"Ingested {n} chunks into the knowledge base.")
