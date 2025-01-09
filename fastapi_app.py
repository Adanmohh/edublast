import os
import tempfile
import uuid
import json
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Query
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# === LlamaIndex / Qdrant / Agents ===
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ----- Global app & in-memory references ------
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize the global index when the server starts"""
    global GLOBAL_INDEX
    GLOBAL_INDEX = _create_or_load_index_from_qdrant()

# Check required environment variables
required_env_vars = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize Qdrant client with cloud configuration
try:
    print(f"Attempting to connect to Qdrant at URL: {os.getenv('QDRANT_URL')}")
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=30.0  # Increased timeout to 30 seconds
    )
    # Test the connection
    print("Testing Qdrant connection...")
    collections = qdrant_client.get_collections()
    print(f"Successfully connected to Qdrant. Available collections: {collections}")
except Exception as e:
    print(f"Error connecting to Qdrant: {str(e)}")
    raise

# We'll store the index name / collection name in a global variable
QDRANT_COLLECTION_NAME = "curriculum_collection"

# Keep references to an Index, QueryEngine, or Agent in memory
GLOBAL_INDEX: Optional[VectorStoreIndex] = None
GLOBAL_AGENT: Optional[ReActAgent] = None

# LlamaIndex Settings: set a custom LLM (e.g., GPT-4 or GPT-3.5-turbo)
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",  # Using a standard OpenAI model
    temperature=0.2
)


# ----- Pydantic Models ------
class LessonContent(BaseModel):
    """Content for a single lesson within a module."""
    title: str = Field(description="The title of this lesson")
    objectives: List[str] = Field(description="Specific learning objectives for this lesson", default_factory=list)
    key_points: List[str] = Field(description="Main concepts and points to be covered", default_factory=list)
    activities: List[str] = Field(description="Practical exercises and learning activities", default_factory=list)
    assessment_ideas: List[str] = Field(description="Methods to assess understanding", default_factory=list)
    resources: List[str] = Field(description="Learning materials and resources needed", default_factory=list)

class ModuleContent(BaseModel):
    """Content for a module containing multiple lessons."""
    title: str = Field(description="The title of this module")
    description: str = Field(description="Overview and goals of this module")
    lessons: List[LessonContent] = Field(description="Individual lessons in this module", default_factory=list)

class CurriculumStructure(BaseModel):
    """Complete curriculum structure with metadata."""
    title: str = Field(description="The main title of the curriculum")
    overview: str = Field(description="Brief overview of the curriculum content")
    target_audience: str = Field(description="Intended audience and prerequisites")
    learning_goals: List[str] = Field(description="Overall learning objectives", default_factory=list)
    modules: List[ModuleContent] = Field(description="Organized modules of content", default_factory=list)


# ===== Helper Functions =====

def _ensure_collection_exists() -> bool:
    """Ensure the Qdrant collection exists, creating it if necessary."""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        exists = any(c.name == QDRANT_COLLECTION_NAME for c in collections.collections)
        
        if exists:
            print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists")
            return True
            
        # Create new collection
        print(f"Creating collection '{QDRANT_COLLECTION_NAME}'...")
        from qdrant_client.http import models
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI embedding size
                distance=models.Distance.COSINE
            )
        )
        
        # Verify collection was created
        collections = qdrant_client.get_collections()
        if any(c.name == QDRANT_COLLECTION_NAME for c in collections.collections):
            print("Collection created successfully")
            return True
        else:
            print("Failed to create collection - not found after creation")
            return False
            
    except Exception as e:
        print(f"Error ensuring collection exists: {str(e)}")
        return False

def _create_or_load_index_from_qdrant() -> Optional[VectorStoreIndex]:
    """Create or load a VectorStoreIndex from Qdrant."""
    try:
        # Ensure collection exists
        if not _ensure_collection_exists():
            print("Failed to create/verify collection")
            return None
            
        print("Loading index from collection...")
        vector_store = QdrantVectorStore(
            client=qdrant_client, 
            collection_name=QDRANT_COLLECTION_NAME
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Try to load existing vectors
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )
            print("Successfully loaded existing index")
        except Exception as e:
            print(f"Creating new empty index: {str(e)}")
            index = VectorStoreIndex([], storage_context=storage_context)
            
        return index
        
    except Exception as e:
        print(f"Error creating/loading index: {str(e)}")
        return None


def _ingest_pdf_to_index(file_path: str, index: VectorStoreIndex, original_filename: str):
    """
    Use LlamaIndex to read the PDF and insert into the VectorStoreIndex.
    """
    try:
        # Load and preprocess documents
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        nodes_to_insert = []
        
        for doc in documents:
            # Clean and validate the text
            if not doc.text or not isinstance(doc.text, str):
                print(f"Skipping invalid document text: {doc.text}")
                continue
                
            # Add metadata to the document
            metadata = {
                "file_name": original_filename,
                "doc_id": str(uuid.uuid4())
            }
            doc.metadata = metadata
            
            # Create a node with metadata
            from llama_index.core import Document
            node = Document(
                text=doc.text.strip(),  # Clean the text
                metadata=metadata,
                excluded_embed_metadata_keys=["file_name", "doc_id"],
                excluded_llm_metadata_keys=["file_name", "doc_id"]
            )
            print(f"Created node with metadata: {metadata}")
            nodes_to_insert.append(node)
        
        # Batch insert all nodes
        if nodes_to_insert:
            print(f"Inserting {len(nodes_to_insert)} nodes into index...")
            index.insert_nodes(nodes_to_insert)
            print("Successfully inserted all nodes")
        else:
            print("No valid nodes to insert")
            
    except Exception as e:
        print(f"Error ingesting PDF: {str(e)}")
        raise Exception(f"Failed to ingest PDF: {str(e)}")

@app.get("/debug_collection")
def debug_collection(collection_name: str = Query(...)):
    """Debug endpoint to inspect collection contents"""
    try:
        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"Collection info: {collection_info}")
        
        # Get all points with payloads
        points = qdrant_client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False
        )[0]
        
        # Extract document info
        documents = []
        seen_files = set()  # Track unique files
        
        for point in points:
            print(f"Point ID: {point.id}")
            print(f"Payload: {point.payload}")
            try:
                if point.payload and '_node_content' in point.payload:
                    # Parse the node_content JSON string
                    node_content = json.loads(point.payload['_node_content'])
                    if 'metadata' in node_content:
                        file_name = node_content['metadata'].get('file_name', 'Unknown')
                        if file_name not in seen_files:  # Only add unique files
                            doc_info = {
                                'id': point.id,
                                'file_name': file_name,
                                'doc_id': node_content['metadata'].get('doc_id', 'Unknown')
                            }
                            documents.append(doc_info)
                            seen_files.add(file_name)
                            print(f"Extracted document info: {doc_info}")
            except json.JSONDecodeError as e:
                print(f"Error parsing node content: {str(e)}")
                continue
        
        return {
            "collection_info": str(collection_info),
            "document_count": len(points),
            "documents": documents
        }
    except Exception as e:
        print(f"Debug error: {str(e)}")
        return {"error": str(e)}


def _build_agent_with_index(index: VectorStoreIndex) -> ReActAgent:
    """
    Build a ReActAgent that uses the vector index as a tool,
    plus any additional tools you may want to add (like generation).
    """
    # Create a QueryEngine from the index
    query_engine = index.as_query_engine()
    
    # Wrap the query engine in a tool
    query_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="CurriculumQueryTool",
        description="Use this to answer questions about the curriculum data."
    )

    # Build the ReActAgent from the single query tool.
    agent = ReActAgent.from_tools([query_tool], verbose=True)
    return agent


def _generate_course_structure(index: VectorStoreIndex) -> CurriculumStructure:
    """Generate a hierarchical course structure from the curriculum content."""
    try:
        # Use structured LLM for better extraction
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2)
        sllm = llm.as_structured_llm(CurriculumStructure)
        
        # Get all text from the index
        query_engine = index.as_query_engine()
        content = query_engine.query(
            "Please provide all the curriculum content in a clear format."
        )
        
        print("Generating course structure...")
        response = sllm.complete(str(content))
        
        if not response.raw:
            raise ValueError("No structure generated")
            
        print(f"Successfully generated curriculum structure: {response.raw.title}")
        return response.raw
        
    except Exception as e:
        print(f"Error generating course structure: {str(e)}")
        return CurriculumStructure(
            title="Error: Generation Failed",
            overview="Failed to generate curriculum structure",
            target_audience="Unknown",
            learning_goals=[],
            modules=[]
        )


# ------ FastAPI Routes ------

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.get("/list_curricula")
def list_curricula():
    """
    List all available Qdrant collections.
    """
    try:
        print("\n=== Listing Collections ===")
        collections = qdrant_client.get_collections()
        collection_list = [c.name for c in collections.collections]
        print(f"Found collections: {collection_list}")
        
        return {
            "collections": collection_list,
            "total_collections": len(collection_list)
        }
        
    except Exception as e:
        error_msg = f"Error listing collections: {str(e)}"
        print(error_msg)
        return {
            "error": error_msg,
            "collections": []
        }


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """
    Ingest a new PDF curriculum file into the Qdrant-backed VectorStoreIndex.
    """
    global GLOBAL_INDEX, GLOBAL_AGENT

    try:
        print(f"\n=== Processing upload for file: {file.filename} ===")
        
        # 1. Save file to temp location
        suffix = file.filename.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            content = file.file.read()
            if not content:
                return {"status": "error", "detail": "Empty file uploaded"}
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
            print(f"Saved file temporarily to: {tmp_path}")

        # 2. Ensure collection exists and load/create index
        if GLOBAL_INDEX is None:
            print("Creating new global index...")
            if not _ensure_collection_exists():
                return {"status": "error", "detail": "Failed to create Qdrant collection"}
                
            GLOBAL_INDEX = _create_or_load_index_from_qdrant()
            if GLOBAL_INDEX is None:
                return {"status": "error", "detail": "Failed to create index"}
        else:
            print("Using existing global index")
            # Still ensure collection exists
            if not _ensure_collection_exists():
                return {"status": "error", "detail": "Failed to verify Qdrant collection"}

        # 3. Ingest PDF into the index with original filename
        print("Ingesting PDF into index...")
        _ingest_pdf_to_index(tmp_path, GLOBAL_INDEX, file.filename)

        # 4. Persist the updated index so we don't have to re-embed
        print("Persisting index...")
        GLOBAL_INDEX.storage_context.persist()

        # 5. Build or rebuild the agent
        print("Building agent...")
        GLOBAL_AGENT = _build_agent_with_index(GLOBAL_INDEX)

        print("Upload process completed successfully")
        return {"status": "success", "detail": "File ingested and index updated."}
        
    except Exception as e:
        error_msg = f"Error processing upload: {str(e)}"
        print(error_msg)
        return {"status": "error", "detail": error_msg}


@app.get("/query")
def query_curriculum(question: str = Query(...)):
    """
    Query the curriculum with a direct question. 
    Uses the ReActAgent if available, otherwise a direct query_engine.
    """
    global GLOBAL_INDEX, GLOBAL_AGENT
    if not GLOBAL_INDEX:
        return {"error": "No index available. Upload a PDF first."}

    if not GLOBAL_AGENT:
        GLOBAL_AGENT = _build_agent_with_index(GLOBAL_INDEX)

    response = GLOBAL_AGENT.chat(question)
    return {"answer": str(response)}


@app.get("/generate_course")
def generate_course(collection_name: str = Query(...)):
    """
    Generate a multi-level course structure from the specified collection.
    Returns JSON (course → modules → lessons).
    """
    try:
        print(f"\n=== Generating course from collection: {collection_name} ===")
        
        # Get all points from the collection
        points = qdrant_client.scroll(
            collection_name=collection_name,
            limit=100,  # Increased limit to get more content
            with_payload=True,
            with_vectors=False
        )[0]

        if not points:
            return {"error": "No content found in collection"}

        print(f"Found {len(points)} documents in collection")
        
        # Create an index with all documents in the collection
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        collection_index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )

        print("Generating course structure...")
        result = _generate_course_structure(collection_index)
        print("Course structure generated successfully")
        
        return result.dict()
        
    except Exception as e:
        error_msg = f"Error generating course: {str(e)}"
        print(error_msg)
        return {"error": error_msg}
