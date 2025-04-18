from setuptools import setup, find_packages

setup(
    name="document-qa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "python-multipart==0.0.6",
        "pydantic==2.5.2",
        "PyMuPDF==1.23.8",
        "python-docx==0.8.11",
        "python-pptx==0.6.21",
        "pandas==2.1.3",
        "openpyxl==3.1.2",
        "pytesseract==0.3.10",
        "Pillow==10.1.0",
        "beautifulsoup4==4.12.2",
        "requests==2.31.0",
        "huggingface-hub==0.16.4",
        "sentence-transformers==2.2.2",
        "faiss-cpu==1.7.4",
        "langchain==0.0.350",
        "python-dotenv==1.0.0",
        "groq==0.3.0",
        "torch==2.1.0",
        "transformers==4.34.0",
    ],
) 