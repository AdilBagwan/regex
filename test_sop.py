from fastapi import FastAPI, HTTPException
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from PyPDF2 import PdfReader
import numpy as np
import re

app = FastAPI()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = "your-azure-openai-api-key"
AZURE_OPENAI_API_BASE = "your-azure-endpoint-url"
AZURE_DEPLOYMENT_EMBEDDING = "text-embedding-ada-002"
AZURE_DEPLOYMENT_CHATGPT = "gpt-4"
AZURE_API_VERSION = "2023-03-15-preview"

# Initialize embeddings and vector database
embeddings = OpenAIEmbeddings(
    model=AZURE_DEPLOYMENT_EMBEDDING,
    deployment=AZURE_DEPLOYMENT_EMBEDDING,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_base=AZURE_OPENAI_API_BASE,
    openai_api_version=AZURE_API_VERSION
)

# Vector store for similarity search
vector_store = None

# SOP content
SOP_CONTENT = {}
SOP_PATH = "path/to/sop.pdf"  # Replace with the actual path

def initialize_sop_and_vectors():
    global SOP_CONTENT, vector_store
    try:
        with open(SOP_PATH, "rb") as f:
            reader = PdfReader(f)
            full_text = " ".join(page.extract_text() for page in reader.pages)

            # Parse sections and subsections
            sections = re.findall(r"(Section \d{2})(.*?)((?=Section \d{2})|$)", full_text, flags=re.S)
            for section, content, _ in sections:
                SOP_CONTENT[section] = {}
                subsections = re.findall(r"(\d\.\d.*?)((?=\d\.\d)|$)", content, flags=re.S)
                for subsection, text in subsections:
                    SOP_CONTENT[section][subsection] = text.strip()

            # Prepare for embeddings
            keys = []
            vector_data = []
            for section, subsections in SOP_CONTENT.items():
                for subsection, text in subsections.items():
                    keys.append(f"{section}-{subsection}")
                    vector_data.append(text)

            # Generate embeddings and create FAISS index
            vectors = embeddings.embed_documents(vector_data)
            vector_store = FAISS(np.array(vectors, dtype="float32"), np.array(keys))

    except FileNotFoundError:
        print("SOP PDF not found. Please ensure the file exists.")
        raise

initialize_sop_and_vectors()

@app.post("/ask")
async def ask_question(query: str):
    """
    Handles free-text queries and returns relevant content from the SOP document.
    """
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized.")

    # Generate query embedding
    query_embedding = embeddings.embed_query(query)

    # Perform similarity search
    distances, indices = vector_store.similarity_search_with_score(query_embedding, k=3)
    matched_sections = [vector_store.reconstruct(idx) for idx in indices]

    if not matched_sections:
        return {"query": query, "response": "No relevant sections found in the document."}

    # Retrieve content for matched sections
    response = "Relevant content from the SOP:\n"
    for match in matched_sections:
        section, subsection = match.split("-", 1)
        content = SOP_CONTENT.get(section, {}).get(subsection, "Content not found.")
        response += f"\n{section} - {subsection}: {content}"

    return {"query": query, "response": response}
