# import requests
# import xml.etree.ElementTree as ET
# import trafilatura
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance, PointStruct
# import cohere

# # -------------------------------------
# # CONFIG
# # -------------------------------------
# # Your Deployment Link:
# SITEMAP_URL = "https://physical-ai-book-six.vercel.app/sitemap.xml"
# COLLECTION_NAME = "humanoid_ai_book"

# cohere_client = cohere.Client("my api key here")
# EMBED_MODEL = "embed-english-v3.0"

# # Connect to Qdrant Cloud
# qdrant = QdrantClient(
#     url="my url here", 
#     api_key="my api key here",
# )

# # -------------------------------------
# # Step 1 — Extract URLs from sitemap
# # -------------------------------------
# def get_all_urls(sitemap_url):
#     xml = requests.get(sitemap_url).text
#     root = ET.fromstring(xml)

#     urls = []
#     for child in root:
#         loc_tag = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
#         if loc_tag is not None:
#             urls.append(loc_tag.text)

#     print("\nFOUND URLS:")
#     for u in urls:
#         print(" -", u)

#     return urls


# # -------------------------------------
# # Step 2 — Download page + extract text
# # -------------------------------------
# def extract_text_from_url(url):
#     html = requests.get(url).text
#     text = trafilatura.extract(html)

#     if not text:
#         print("[WARNING] No text extracted from:", url)

#     return text


# # -------------------------------------
# # Step 3 — Chunk the text
# # -------------------------------------
# def chunk_text(text, max_chars=1200):
#     chunks = []
#     while len(text) > max_chars:
#         split_pos = text[:max_chars].rfind(". ")
#         if split_pos == -1:
#             split_pos = max_chars
#         chunks.append(text[:split_pos])
#         text = text[split_pos:]
#     chunks.append(text)
#     return chunks


# # -------------------------------------
# # Step 4 — Create embedding
# # -------------------------------------
# def embed(text):
#     response = cohere_client.embed(
#         model=EMBED_MODEL,
#         input_type="search_query",  # Use search_query for queries
#         texts=[text],
#     )
#     return response.embeddings[0]  # Return the first embedding


# # -------------------------------------
# # Step 5 — Store in Qdrant
# # -------------------------------------
# def create_collection():
#     print("\nCreating Qdrant collection...")
#     qdrant.recreate_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(
#         size=1024,        # Cohere embed-english-v3.0 dimension
#         distance=Distance.COSINE
#         )
#     )

# def save_chunk_to_qdrant(chunk, chunk_id, url):
#     vector = embed(chunk)

#     qdrant.upsert(
#         collection_name=COLLECTION_NAME,
#         points=[
#             PointStruct(
#                 id=chunk_id,
#                 vector=vector,
#                 payload={
#                     "url": url,
#                     "text": chunk,
#                     "chunk_id": chunk_id
#                 }
#             )
#         ]
#     )


# # -------------------------------------
# # MAIN INGESTION PIPELINE
# # -------------------------------------
# def ingest_book():
#     urls = get_all_urls(SITEMAP_URL)

#     create_collection()

#     global_id = 1

#     for url in urls:
#         print("\nProcessing:", url)
#         text = extract_text_from_url(url)

#         if not text:
#             continue

#         chunks = chunk_text(text)

#         for ch in chunks:
#             save_chunk_to_qdrant(ch, global_id, url)
#             print(f"Saved chunk {global_id}")
#             global_id += 1

#     print("\n✔️ Ingestion completed!")
#     print("Total chunks stored:", global_id - 1)


# if __name__ == "__main__":
#     ingest_book()

# import requests
# import xml.etree.ElementTree as ET
# import trafilatura
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance, PointStruct
# import cohere

# # 🔽 NEW IMPORTS
# from fastapi import FastAPI
# from pydantic import BaseModel
# from agents import Agent, Runner, function_tool
# import sys
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # -------------------------------------
# # CONFIG
# # -------------------------------------
# SITEMAP_URL = "https://physical-ai-book-six.vercel.app/sitemap.xml"
# COLLECTION_NAME = "humanoid_ai_book"

# cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
# EMBED_MODEL = "embed-english-v3.0"

# qdrant = QdrantClient(
#     url=os.getenv("QDRANT_URL"), 
#     api_key=os.getenv("QDRANT_API_KEY"),
# )

# # -------------------------------------
# # INGESTION FUNCTIONS (UNCHANGED)
# # -------------------------------------
# def get_all_urls(sitemap_url):
#     xml = requests.get(sitemap_url).text
#     root = ET.fromstring(xml)

#     urls = []
#     for child in root:
#         loc_tag = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
#         if loc_tag is not None:
#             urls.append(loc_tag.text)

#     return urls


# def extract_text_from_url(url):
#     html = requests.get(url).text
#     return trafilatura.extract(html)


# def chunk_text(text, max_chars=1200):
#     chunks = []
#     while len(text) > max_chars:
#         split_pos = text[:max_chars].rfind(". ")
#         if split_pos == -1:
#             split_pos = max_chars
#         chunks.append(text[:split_pos])
#         text = text[split_pos:]
#     chunks.append(text)
#     return chunks


# def embed(text):
#     response = cohere_client.embed(
#         model=EMBED_MODEL,
#         input_type="search_query",
#         texts=[text],
#     )
#     return response.embeddings[0]


# def create_collection():
#     qdrant.recreate_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(
#             size=1024,
#             distance=Distance.COSINE
#         )
#     )


# def save_chunk_to_qdrant(chunk, chunk_id, url):
#     vector = embed(chunk)

#     qdrant.upsert(
#         collection_name=COLLECTION_NAME,
#         points=[
#             PointStruct(
#                 id=chunk_id,
#                 vector=vector,
#                 payload={
#                     "url": url,
#                     "text": chunk,
#                     "chunk_id": chunk_id
#                 }
#             )
#         ]
#     )


# def ingest_book():
#     urls = get_all_urls(SITEMAP_URL)
#     create_collection()

#     global_id = 1
#     for url in urls:
#         text = extract_text_from_url(url)
#         if not text:
#             continue

#         for ch in chunk_text(text):
#             save_chunk_to_qdrant(ch, global_id, url)
#             global_id += 1

#     print("✔ Ingestion completed")


# # -------------------------------------
# # 🔥 RAG + FASTAPI (ADDED)
# # -------------------------------------

# def get_query_embedding(text: str):
#     return embed(text)


# @function_tool
# def retrieve(query: str):
#     embedding = get_query_embedding(query)
#     result = qdrant.query_points(
#         collection_name=COLLECTION_NAME,
#         query=embedding,
#         limit=5,
#     )
#     return [p.payload["text"] for p in result.points]


# agent = Agent(
#     name="Assistant",
#     instructions="""
# You are an AI tutor for the Physical AI & Humanoid Robotics textbook.

# Steps:
# 1. Call `retrieve` with the user question

# be concise 
# """,
#     model="gemini-flash-lite-latest",   # or your Gemini/OpenRouter model
#     tools=[retrieve],
# )


# app = FastAPI()

# class Query(BaseModel):
#     question: str


# @app.post("/chat")
# def chat(q: Query):
#     result = Runner.run_sync(agent, input=q.question)
#     return {"answer": result.final_output}


# # -------------------------------------
# # ENTRY POINT
# # -------------------------------------
# if __name__ == "__main__":
#     # Run ingestion ONLY when explicitly asked
#     # python main.py ingest
#     if len(sys.argv) > 1 and sys.argv[1] == "ingest":
#         ingest_book()

# import requests
# import xml.etree.ElementTree as ET
# import trafilatura
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance, PointStruct
# import cohere

# # 🔽 NEW IMPORTS
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from agents import Agent, Runner, function_tool
# from agents.models.openai_provider import OpenAIChatCompletionsModel
# import sys
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # -------------------------------------
# # CONFIG
# # -------------------------------------
# SITEMAP_URL = "https://physical-ai-book-six.vercel.app/sitemap.xml"
# COLLECTION_NAME = "humanoid_ai_book"

# cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
# EMBED_MODEL = "embed-english-v3.0"

# qdrant = QdrantClient(
#     url=os.getenv("QDRANT_URL"), 
#     api_key=os.getenv("QDRANT_API_KEY"),
# )

# GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# # -------------------------------------
# # INGESTION FUNCTIONS
# # -------------------------------------
# def get_all_urls(sitemap_url):
#     xml = requests.get(sitemap_url).text
#     root = ET.fromstring(xml)
#     urls = []
#     for child in root:
#         loc_tag = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
#         if loc_tag is not None:
#             urls.append(loc_tag.text)
#     return urls

# def extract_text_from_url(url):
#     html = requests.get(url).text
#     return trafilatura.extract(html)

# def chunk_text(text, max_chars=1200):
#     chunks = []
#     while len(text) > max_chars:
#         split_pos = text[:max_chars].rfind(". ")
#         if split_pos == -1:
#             split_pos = max_chars
#         chunks.append(text[:split_pos])
#         text = text[split_pos:]
#     chunks.append(text)
#     return chunks

# def embed(text):
#     response = cohere_client.embed(
#         model=EMBED_MODEL,
#         input_type="search_query",
#         texts=[text],
#     )
#     return response.embeddings[0]

# def create_collection():
#     qdrant.recreate_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(
#             size=1024,
#             distance=Distance.COSINE
#         )
#     )

# def save_chunk_to_qdrant(chunk, chunk_id, url):
#     vector = embed(chunk)
#     qdrant.upsert(
#         collection_name=COLLECTION_NAME,
#         points=[PointStruct(
#             id=chunk_id,
#             vector=vector,
#             payload={
#                 "url": url,
#                 "text": chunk,
#                 "chunk_id": chunk_id
#             }
#         )]
#     )

# def ingest_book():
#     urls = get_all_urls(SITEMAP_URL)
#     create_collection()
#     global_id = 1
#     for url in urls:
#         text = extract_text_from_url(url)
#         if not text:
#             continue
#         for ch in chunk_text(text):
#             save_chunk_to_qdrant(ch, global_id, url)
#             global_id += 1
#     print("✔ Ingestion completed")

# # -------------------------------------
# # RAG + FASTAPI
# # -------------------------------------
# def get_query_embedding(text: str):
#     return embed(text)

# @function_tool
# def retrieve(query: str):
#     embedding = get_query_embedding(query)
#     result = qdrant.query_points(
#         collection_name=COLLECTION_NAME,
#         query=embedding,
#         limit=5,
#     )
#     return [p.payload["text"] for p in result.points]

# # Explicitly use OpenAI-compatible model for Gemini
# gemini_model = OpenAIChatCompletionsModel(
#     "gemini-flash-lite-latest",
#     api_key=os.getenv("GEMINI_API_KEY")
# )

# agent = Agent(
#     name="Assistant",
#     instructions="""
# You are an AI tutor for the Physical AI & Humanoid Robotics textbook.

# Steps:
# 1. Call `retrieve` with the user question.
# Be concise.
# """,
#     model=gemini_model,
#     tools=[retrieve],
# )

# # FastAPI app
# app = FastAPI()

# # Enable CORS for Docusaurus frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change to your frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class Query(BaseModel):
#     question: str

# @app.post("/chat")
# def chat(q: Query):
#     try:
#         result = Runner.run_sync(agent, input=q.question)
#         return {"answer": result.final_output}
#     except Exception as e:
#         return {"error": str(e)}

# # -------------------------------------
# # ENTRY POINT
# # -------------------------------------
# if __name__ == "__main__":
#     # Run ingestion ONLY when explicitly asked
#     # python main.py ingest
#     if len(sys.argv) > 1 and sys.argv[1] == "ingest":
#         ingest_book()


# --------------------------------------------------------------------------------------------------------
# import requests
# import xml.etree.ElementTree as ET
# import trafilatura
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance, PointStruct
# import cohere

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from agents import Agent, Runner, function_tool
# from agents.models.openai_provider import OpenAIChatCompletionsModel, AsyncOpenAI
# import sys
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # -------------------------------------
# # CONFIG
# # -------------------------------------
# SITEMAP_URL = "https://physical-ai-book-six.vercel.app/sitemap.xml"
# COLLECTION_NAME = "humanoid_ai_book"

# cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
# EMBED_MODEL = "embed-english-v3.0"

# qdrant = QdrantClient(
#     url=os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY"),
# )

# # Ensure GEMINI API key is set
# gemini_key = os.getenv("GEMINI_API_KEY")
# if not gemini_key:
#     raise ValueError("GEMINI_API_KEY not set in .env")

# # Set environment variable for OpenAI-compatible client

# # -------------------------------------
# # INGESTION FUNCTIONS
# # -------------------------------------
# def get_all_urls(sitemap_url):
#     xml = requests.get(sitemap_url).text
#     root = ET.fromstring(xml)
#     urls = []
#     for child in root:
#         loc_tag = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
#         if loc_tag is not None:
#             urls.append(loc_tag.text)
#     return urls

# def extract_text_from_url(url):
#     html = requests.get(url).text
#     return trafilatura.extract(html)

# def chunk_text(text, max_chars=1200):
#     chunks = []
#     while len(text) > max_chars:
#         split_pos = text[:max_chars].rfind(". ")
#         if split_pos == -1:
#             split_pos = max_chars
#         chunks.append(text[:split_pos])
#         text = text[split_pos:]
#     chunks.append(text)
#     return chunks

# def embed(text):
#     response = cohere_client.embed(
#         model=EMBED_MODEL,
#         input_type="search_query",
#         texts=[text],
#     )
#     return response.embeddings[0]

# def create_collection():
#     qdrant.recreate_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(
#             size=1024,
#             distance=Distance.COSINE
#         )
#     )

# def save_chunk_to_qdrant(chunk, chunk_id, url):
#     vector = embed(chunk)
#     qdrant.upsert(
#         collection_name=COLLECTION_NAME,
#         points=[PointStruct(
#             id=chunk_id,
#             vector=vector,
#             payload={
#                 "url": url,
#                 "text": chunk,
#                 "chunk_id": chunk_id
#             }
#         )]
#     )

# def ingest_book():
#     urls = get_all_urls(SITEMAP_URL)
#     create_collection()
#     global_id = 1
#     for url in urls:
#         text = extract_text_from_url(url)
#         if not text:
#             continue
#         for ch in chunk_text(text):
#             save_chunk_to_qdrant(ch, global_id, url)
#             global_id += 1
#     print("✔ Ingestion completed")

# # -------------------------------------
# # RAG + FASTAPI
# # -------------------------------------
# def get_query_embedding(text: str):
#     return embed(text)

# @function_tool
# def retrieve(query: str):
#     embedding = get_query_embedding(query)
#     result = qdrant.query_points(
#         collection_name=COLLECTION_NAME,
#         query=embedding,
#         limit=5,
#     )
#     return [p.payload["text"] for p in result.points]

# # Initialize OpenAI-compatible async client for Gemini
# openai_client = AsyncOpenAI(
#     api_key=gemini_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )
#   # automatically reads OPENAI_API_KEY

# gemini_model = OpenAIChatCompletionsModel(
#     model="gemini-flash-lite-latest",
#     openai_client=openai_client
# )

# # Create agent
# agent = Agent(
#     name="Assistant",
#     instructions="""
# You are an AI tutor for the Physical AI & Humanoid Robotics textbook.

# Steps:
# 1. Call `retrieve` with the user question.
# Be concise.
# """,
#     model=gemini_model,
#     tools=[retrieve],
# )

# # FastAPI app
# app = FastAPI()

# # Enable CORS for frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace with frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class Query(BaseModel):
#     question: str

# @app.post("/chat")
# def chat(q: Query):
#     try:
#         result = Runner.run_sync(agent, input=q.question)
#         return {"answer": result.final_output}
#     except Exception as e:
#         return {"error": str(e)}

# # -------------------------------------
# # ENTRY POINT
# # -------------------------------------
# if __name__ == "__main__":
#     if len(sys.argv) > 1 and sys.argv[1] == "ingest":
#         ingest_book()
# ----------------------------------------------------------------------------------------------------------

import requests
import xml.etree.ElementTree as ET
import trafilatura
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import cohere

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import Agent, Runner, function_tool
from agents.models.openai_provider import OpenAIChatCompletionsModel, AsyncOpenAI
import sys
import os
from dotenv import load_dotenv

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()

SITEMAP_URL = "https://physical-ai-book-six.vercel.app/sitemap.xml"
COLLECTION_NAME = "humanoid_ai_book"
EMBED_MODEL = "embed-english-v3.0"

# -------------------------------------------------
# LAZY CLIENTS (Railway-safe)
# -------------------------------------------------
_cohere_client = None
_qdrant_client = None

def get_cohere():
    global _cohere_client
    if _cohere_client is None:
        key = os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY not set")
        _cohere_client = cohere.Client(key)
    return _cohere_client

def get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
    return _qdrant_client

# -------------------------------------------------
# INGESTION
# -------------------------------------------------
def get_all_urls(sitemap_url):
    xml = requests.get(sitemap_url).text
    root = ET.fromstring(xml)
    urls = []
    for child in root:
        loc = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
        if loc is not None:
            urls.append(loc.text)
    return urls

def extract_text_from_url(url):
    html = requests.get(url).text
    return trafilatura.extract(html)

def chunk_text(text, max_chars=1200):
    chunks = []
    while len(text) > max_chars:
        split = text[:max_chars].rfind(". ")
        split = split if split != -1 else max_chars
        chunks.append(text[:split])
        text = text[split:]
    chunks.append(text)
    return chunks

def embed(text):
    client = get_cohere()
    res = client.embed(
        model=EMBED_MODEL,
        input_type="search_query",
        texts=[text],
    )
    return res.embeddings[0]

def create_collection():
    get_qdrant().recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

def save_chunk_to_qdrant(chunk, chunk_id, url):
    get_qdrant().upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=chunk_id,
                vector=embed(chunk),
                payload={"url": url, "text": chunk},
            )
        ],
    )

def ingest_book():
    create_collection()
    gid = 1
    for url in get_all_urls(SITEMAP_URL):
        text = extract_text_from_url(url)
        if not text:
            continue
        for ch in chunk_text(text):
            save_chunk_to_qdrant(ch, gid, url)
            gid += 1
    print("✔ Ingestion completed")

# -------------------------------------------------
# RAG TOOL
# -------------------------------------------------
@function_tool
def retrieve(query: str):
    vec = embed(query)
    res = get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=5,
    )
    return [p.payload["text"] for p in res.points]

# -------------------------------------------------
# AGENT (UNCHANGED)
# -------------------------------------------------
openai_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-flash-lite-latest",
    openai_client=openai_client,
)

agent = Agent(
    name="Assistant",
    instructions="""
You are an AI tutor for the Physical AI & Humanoid Robotics textbook.

Steps:
1. Call `retrieve` with the user question.
Be concise.
""",
    model=gemini_model,
    tools=[retrieve],
)

# -------------------------------------------------
# FASTAPI
# -------------------------------------------------
app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Query):
    result = Runner.run_sync(agent, input=q.question)
    return {"answer": result.final_output}

# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
# if __name__ == "__main__":
#     if len(sys.argv) > 1 and sys.argv[1] == "ingest":
#         ingest_book()
if __name__ == "__main__":
    import uvicorn
    import os

    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest_book()
    else:
        # Use Railway's port or default to 8000 locally
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
