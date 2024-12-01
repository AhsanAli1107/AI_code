#main temp
import os
import getpass
# import nest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SimpleNodeParser, TokenTextSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from llama_index.core.retrievers import VectorIndexRetriever
import streamlit as st
import requests


# nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI()

# Define the request model
class QueryRequest(BaseModel):
    query: str

# Initialize the embedding model and LLM
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
ollama_llm = Ollama(model="llama3", base_url='http://localhost:11434', request_timeout=360.0)

# Set chunking settings
Settings.chunk_size = 1024
Settings.chunk_overlap = 50

# Initialize ChromaDB and vector store
db = chromadb.PersistentClient(path="./chroma_db/test")
chroma_collection = db.get_or_create_collection("test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load documents and create index
documents = SimpleDirectoryReader(input_dir='book', filename_as_id=True).load_data()
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model, show_progress=True)

# Initialize retriever and query engine
retriever = VectorIndexRetriever(index=index)
query_engine = index.as_query_engine(llm=ollama_llm,  similarity_top_k=5,)

@app.post("/query")
async def query_chatbot(query_request: QueryRequest):
    try:
        # Get the query from the request
        query = query_request.query

        # Query the engine
        response = query_engine.query(query)
        
        # Return the response
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    