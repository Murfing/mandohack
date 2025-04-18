import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
from pathlib import Path
from app.document_processor import DocumentProcessor
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Document QA Application",
    description="A FastAPI application for document question answering",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the static directory
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize document processor
processor = DocumentProcessor()

class Query(BaseModel):
    question: str
    mode: str = "simple"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Handle file uploads and process documents."""
    try:
        # Clear previous uploads
        for file in UPLOAD_DIR.glob("*"):
            file.unlink()
            
        # Save new files
        for file in files:
            file_path = UPLOAD_DIR / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        # Process documents
        processor.process_documents(UPLOAD_DIR)
        
        return {
            "message": "Files uploaded and processed successfully",
            "files": [file.filename for file in files]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: Query):
    """Handle document queries."""
    try:
        if not processor.is_initialized():
            raise HTTPException(status_code=400, detail="No documents uploaded yet")
            
        result = processor.query(query.question, query.mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    ) 