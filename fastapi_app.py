import os
import tempfile
import uuid
import json
from typing import List, Optional, Dict
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Query, Body
from pydantic import BaseModel, Field

# === LlamaIndex / Qdrant / Agents ===
import llama_index.vector_stores.qdrant
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# === Load environment variables from .env file ===
load_dotenv()

app = FastAPI()

# ----- Global & Config -----
QDRANT_COLLECTION_NAME = "curriculum_collection"
GLOBAL_INDEX: Optional[VectorStoreIndex] = None
GLOBAL_AGENT: Optional[ReActAgent] = None

# In-memory storage for course outlines and lessons (for demo)
# Key = outline_id, Value = JSON/dict representing the outline
OUTLINE_DB: Dict[str, dict] = {}

required_env_vars = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize Qdrant client
try:
    print(f"Attempting to connect to Qdrant at URL: {os.getenv('QDRANT_URL')}")
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=30.0
    )
    print("Testing Qdrant connection...")
    collections = qdrant_client.get_collections()
    print(f"Successfully connected to Qdrant. Collections: {collections}")
except Exception as e:
    print(f"Error connecting to Qdrant: {str(e)}")
    raise

# LlamaIndex Settings: use GPT-4 (adjust as needed)
Settings.llm = OpenAI(model="gpt-4", temperature=0.2)


# ====== Pydantic Models for Curriculum ======
class LessonContent(BaseModel):
    title: str = Field(description="The title of this lesson")
    objectives: List[str] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)
    assessment_ideas: List[str] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)

class ModuleContent(BaseModel):
    title: str
    description: str
    lessons: List[LessonContent] = Field(default_factory=list)

class CurriculumStructure(BaseModel):
    title: str
    overview: str
    target_audience: str
    learning_goals: List[str] = Field(default_factory=list)
    modules: List[ModuleContent] = Field(default_factory=list)


# ===== Helper Functions for Qdrant & Index =====
def _ensure_collection_exists() -> bool:
    try:
        collections = qdrant_client.get_collections()
        exists = any(c.name == QDRANT_COLLECTION_NAME for c in collections.collections)
        if exists:
            print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists")
            return True
        
        print(f"Creating collection '{QDRANT_COLLECTION_NAME}'...")
        from qdrant_client.http import models
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE
            )
        )
        # Verify
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
    try:
        if not _ensure_collection_exists():
            print("Failed to create/verify Qdrant collection")
            return None
        
        print("Loading index from collection...")
        vector_store = QdrantVectorStore(client=qdrant_client, collection_name=QDRANT_COLLECTION_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        try:
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            print("Successfully loaded existing index")
        except Exception as e:
            print(f"Creating new empty index: {str(e)}")
            index = VectorStoreIndex([], storage_context=storage_context)
        return index
    except Exception as e:
        print(f"Error creating/loading index: {str(e)}")
        return None

def _ingest_pdf_to_index(file_path: str, index: VectorStoreIndex, original_filename: str):
    try:
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        nodes_to_insert = []
        
        for doc in docs:
            if not doc.text or not isinstance(doc.text, str):
                print(f"Skipping invalid doc text: {doc.text}")
                continue
            metadata = {"file_name": original_filename, "doc_id": str(uuid.uuid4())}
            doc.metadata = metadata

            from llama_index.core import Document
            node = Document(
                text=doc.text.strip(),
                metadata=metadata,
                excluded_embed_metadata_keys=["file_name","doc_id"],
                excluded_llm_metadata_keys=["file_name","doc_id"],
            )
            nodes_to_insert.append(node)
        
        if nodes_to_insert:
            index.insert_nodes(nodes_to_insert)
            print(f"Inserted {len(nodes_to_insert)} nodes.")
        else:
            print("No valid nodes to insert from PDF.")
    except Exception as e:
        raise Exception(f"Failed to ingest PDF: {str(e)}")


def _build_agent_with_index(index: VectorStoreIndex) -> ReActAgent:
    query_engine = index.as_query_engine()
    query_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="CurriculumQueryTool",
        description="Use this to answer questions about the ingested docs."
    )
    agent = ReActAgent.from_tools([query_tool], verbose=True)
    return agent


# ===== Startup =====
@app.on_event("startup")
async def startup_event():
    global GLOBAL_INDEX, GLOBAL_AGENT
    print("Starting up app...")
    GLOBAL_INDEX = _create_or_load_index_from_qdrant()
    if GLOBAL_INDEX:
        GLOBAL_AGENT = _build_agent_with_index(GLOBAL_INDEX)


# ===== FastAPI Endpoints =====

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.get("/list_curricula")
def list_curricula():
    """List all Qdrant collections."""
    try:
        collections = qdrant_client.get_collections()
        collection_list = [c.name for c in collections.collections]
        return {"status": "success", "collections": collection_list}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """
    Ingest a new PDF into Qdrant and rebuild the agent.
    """
    global GLOBAL_INDEX, GLOBAL_AGENT
    try:
        suffix = file.filename.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            content = file.file.read()
            if not content:
                return {"status": "error", "detail": "Empty PDF"}
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
        
        # Ensure we have an index
        if GLOBAL_INDEX is None:
            if not _ensure_collection_exists():
                return {"status": "error", "detail": "Could not create collection"}
            GLOBAL_INDEX = _create_or_load_index_from_qdrant()
            if not GLOBAL_INDEX:
                return {"status": "error", "detail": "Could not init index"}
        
        # Ingest
        _ingest_pdf_to_index(tmp_path, GLOBAL_INDEX, file.filename)
        GLOBAL_INDEX.storage_context.persist()

        # Rebuild agent
        GLOBAL_AGENT = _build_agent_with_index(GLOBAL_INDEX)

        return {"status": "success", "detail": f"PDF '{file.filename}' ingested."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/query")
def query_curriculum(question: str):
    """
    Ask a freeform question. The ReActAgent can fetch data from Qdrant if needed.
    """
    global GLOBAL_AGENT
    if not GLOBAL_AGENT:
        return {"status": "error", "detail": "Agent not initialized"}
    response = GLOBAL_AGENT.chat(question)
    return {"status": "success", "answer": str(response)}


# === Step-by-Step Course Creation Endpoints ===

@app.post("/create_course_outline")
def create_course_outline(
    title: str = Body(...),
    short_desc: str = Body(...),
    duration_weeks: int = Body(...),
    curriculum: str = Body(default="General Knowledge"),
):
    """
    Creates a high-level course outline in JSON format.
    Returns an outline_id that can be used for further steps.
    """
    global GLOBAL_AGENT
    if not GLOBAL_AGENT:
        return {"status": "error", "detail": "Agent not initialized. Upload a PDF first if needed."}

    outline_prompt = (
        "You are an instructional designer. Generate a JSON outline for a course:\n\n"
        f"Title: {title}\n"
        f"Description: {short_desc}\n"
        f"Duration: {duration_weeks} weeks (each week = 1 module).\n\n"
        "Include for each module:\n"
        "- module_title\n"
        "- module_summary\n"
        "- lesson_topics (a short list)\n\n"
        "Return JSON ONLY, e.g.:\n"
        "{\n"
        "  \"course_title\": \"...\",\n"
        "  \"modules\": [\n"
        "    {\n"
        "      \"module_title\": \"Week 1: ...\",\n"
        "      \"module_summary\": \"...\",\n"
        "      \"lesson_topics\": [\"...\", \"...\"]\n"
        "    }...\n"
        "  ]\n"
        "}\n"
    )

    resp = GLOBAL_AGENT.chat(outline_prompt)
    outline_str = str(resp)
    outline_id = str(uuid.uuid4())

    # Attempt to parse JSON
    try:
        outline_data = json.loads(outline_str)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "detail": "Failed to parse JSON from agent. Try adjusting prompt.",
            "raw_response": outline_str
        }

    # Store in memory
    OUTLINE_DB[outline_id] = outline_data
    # Set source based on selected curriculum
    source = f"Curriculum: {curriculum}" if curriculum != "General Knowledge" else "General Knowledge"
    
    return {
        "status": "success",
        "data": {
            "outline_id": outline_id,
            "outline": outline_data,
            "source": source
        }
    }


@app.post("/approve_outline")
def approve_outline(
    outline_id: str = Body(...),
    updated_outline: dict = Body(...)
):
    """
    User can modify the returned outline if needed, then approve it.
    We'll store the updated version in memory.
    """
    if outline_id not in OUTLINE_DB:
        return {"status": "error", "detail": "Outline ID not found"}
    OUTLINE_DB[outline_id] = updated_outline
    return {"status": "success", "final_outline": updated_outline}


@app.post("/generate_lesson")
def generate_lesson(
    outline_id: str = Body(...),
    module_index: int = Body(...),
    lesson_prompt: str = Body(...),
):
    """
    Generate exactly ONE lesson in JSON form, appended to the specified module.
    Assumes user has an existing 'outline' with modules.
    """
    global GLOBAL_AGENT
    if not GLOBAL_AGENT:
        return {"status": "error", "detail": "Agent not initialized"}

    if outline_id not in OUTLINE_DB:
        return {"status": "error", "detail": "Invalid outline_id"}
    outline_data = OUTLINE_DB[outline_id]

    # Grab the module info
    modules = outline_data.get("modules", [])
    if module_index < 0 or module_index >= len(modules):
        return {"status": "error", "detail": "module_index out of range"}
    module_info = modules[module_index]

    # Prompt the agent
    prompt = (
        f"We have this outline:\n{json.dumps(outline_data, indent=2)}\n\n"
        f"Focus on module index {module_index} titled '{module_info.get('module_title')}'.\n"
        "Generate exactly ONE lesson in JSON with keys:\n"
        "title, objectives, key_points, activities, assessment_ideas, resources.\n"
        "Lesson requirements:\n"
        f"{lesson_prompt}\n\n"
        "Return ONLY JSON. Example:\n"
        "{\n"
        "  \"title\": \"...\",\n"
        "  \"objectives\": [...],\n"
        "  \"key_points\": [...],\n"
        "  \"activities\": [...],\n"
        "  \"assessment_ideas\": [...],\n"
        "  \"resources\": [...]\n"
        "}\n"
    )

    resp = GLOBAL_AGENT.chat(prompt)
    lesson_str = str(resp)

    # Parse the lesson
    try:
        lesson_data = json.loads(lesson_str)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "detail": "Failed to parse JSON for lesson.",
            "raw_response": lesson_str
        }

    # Append to the outline's module
    if "lessons" not in module_info:
        module_info["lessons"] = []
    module_info["lessons"].append(lesson_data)

    # Save back to DB
    modules[module_index] = module_info
    outline_data["modules"] = modules
    OUTLINE_DB[outline_id] = outline_data

    return {"status": "success", "lesson": lesson_data, "updated_outline": outline_data}


@app.get("/propose_lesson_prompt")
def propose_lesson_prompt(outline_id: str, module_number: int):
    """
    Generate a short 'proposed lesson idea' for a given module (1-based).
    Returns something like:
      { "proposed_idea": "Cover advanced OOP with hands-on examples." }
    """
    global GLOBAL_AGENT
    if not GLOBAL_AGENT:
        return {"status": "error", "detail": "Agent not initialized"}

    if outline_id not in OUTLINE_DB:
        return {"status": "error", "detail": "Invalid outline_id"}

    # Convert module_number to 0-based index
    mod_idx = module_number - 1
    outline_data = OUTLINE_DB[outline_id]
    modules = outline_data.get("modules", [])
    if mod_idx < 0 or mod_idx >= len(modules):
        return {"status": "error", "detail": "module_number out of range"}

    module_title = modules[mod_idx].get("module_title", "Untitled Module")

    # Build a prompt to propose a single-sentence lesson idea
    proposal_prompt = (
        f"You are an expert instructional designer. Suggest one short lesson idea for the "
        f"module '{module_title}'. Provide just one concise sentence describing the lesson focus."
    )

    resp = GLOBAL_AGENT.chat(proposal_prompt)
    proposed_idea = str(resp).strip()

    return {"status": "success", "proposed_idea": proposed_idea}


@app.get("/debug_collection")
def debug_collection(collection_name: str):
    """
    Debug: Inspect Qdrant collection contents.
    """
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        points = qdrant_client.scroll(collection_name=collection_name, limit=100, with_payload=True, with_vectors=False)[0]
        documents = []
        seen_files = set()
        for point in points:
            payload = point.payload
            if payload and "_node_content" in payload:
                node_content = json.loads(payload["_node_content"])
                fname = node_content["metadata"].get("file_name", "Unknown")
                if fname not in seen_files:
                    seen_files.add(fname)
                    documents.append({"id": point.id, "file_name": fname})
        return {
            "status": "success",
            "collection_info": str(collection_info),
            "num_points": len(points),
            "documents": documents
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# Optional: If you want a “generate_course” from all docs
# We keep your original function for reference.
@app.get("/generate_course")
def generate_course(collection_name: str):
    """
    Example that attempts to build a CurriculumStructure from all docs in a given collection.
    """
    from llama_index.core import StorageContext
    try:
        points = qdrant_client.scroll(collection_name=collection_name, limit=200, with_payload=True, with_vectors=False)[0]
        if not points:
            return {"status": "error", "detail": "No content found in collection"}

        vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        collection_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

        # If you want a structured extraction:
        from llama_index.llms.openai import OpenAI
        from llama_index.core.prompts import PromptTemplate
        from llama_index.core import Document
        from llama_index.core.output_parsers import PydanticOutputParser
        from llama_index.core.program import LLMTextCompletionProgram

        # Or do something simpler with an agent or a structured LLM.
        # For now, let's just do something trivial:
        try:
            llm = OpenAI(model="gpt-4", temperature=0.2)
            sllm = llm.as_structured_llm(CurriculumStructure)

            query_engine = collection_index.as_query_engine()
            content = query_engine.query("Please provide all the curriculum content in a clear format.")
            response = sllm.complete(str(content))
            if not response.raw:
                raise ValueError("No structure generated")
            return {"status": "success", "data": response.raw.dict()}
        except Exception as e:
            return {
                "status": "error",
                "detail": str(e),
                "fallback": "No structured data could be generated."
            }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
