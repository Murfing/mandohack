# Document QA Application

A FastAPI-based application for document question answering.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the application and its dependencies:
```bash
pip install -e .
```

## Configuration

Create a `.env` file in the root directory with the following variables:
```
HOST=0.0.0.0
PORT=8000
```

## Running the Application

To start the application, run:
```bash
python run.py
```

The application will be available at `http://localhost:8000`.

## API Documentation

Once the application is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Features

- Support for multiple document formats:
  - PDF (including scanned PDFs with OCR)
  - DOCX
  - PPTX
  - XLSX/CSV
  - JSON
  - TXT
  - Images (PNG, JPG)
- Semantic search using embeddings
- Structured data processing
- Natural language question answering using Groq's Mixtral model
- Reference highlighting and source tracking
- Multiple answer support and summarization

## Architecture

The system consists of three main components:

1. Document Processor: Handles different file formats and extracts content
2. Text Processor: Chunks text and generates embeddings for semantic search
3. Query Processor: Processes questions and generates answers using Groq's Mixtral model

## License

MIT 