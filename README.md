# Document QA Assistant

A modern web application that allows users to upload documents and ask questions about their content. The application uses advanced natural language processing to provide accurate answers and relevant sources from the uploaded documents.

## Features

- **Modern User Interface**
  - Clean, responsive design with smooth animations
  - Intuitive drag-and-drop file upload
  - Real-time file upload status indicators
  - Visual feedback for success and error states
  - Disabled query section until files are uploaded
  - Mobile-friendly layout

- **Document Processing**
  - Support for multiple file formats (PDF, DOCX, TXT)
  - File size validation
  - Real-time upload progress tracking
  - Success/error notifications for file processing

- **Query System**
  - Multiple query modes (default, precise, creative)
  - Disabled query input until files are processed
  - Clear error messages for invalid queries
  - Loading states during query processing

- **Results Display**
  - Well-formatted answers with sources
  - Smooth animations for results display
  - Clear source attribution
  - Error handling with user-friendly messages

## Installation

1. **Prerequisites**
   - Python 3.8 or higher
   - pip (Python package installer)
   - Git
   - A Groq API key (get it from [Groq Cloud](https://console.groq.com/))

2. **Clone the Repository**
```bash
git clone [repository-url]
cd document-qa-assistant
```

3. **Set Up Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

4. **Install Dependencies**
```bash
# Install Python packages
pip install -r requirements.txt

# Install additional system dependencies
# On Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# On macOS:
brew install tesseract
# On Windows:
# Download and install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki
```

5. **Configure API Keys**
   - Create a `.env` file in the root directory
   - Add your Groq API key:
```bash
GROQ_API_KEY=your_api_key_here
```

6. **Run the Application**
```bash
# Start the Flask server
python app.py

# The application will be available at http://localhost:5000
```

7. **Verify Installation**
   - Open your browser and navigate to `http://localhost:5000`
   - You should see the Document QA Assistant interface
   - Try uploading a document to verify the installation

## Requirements

The application requires the following Python packages (automatically installed via requirements.txt):

- Flask==2.0.1
- python-dotenv==0.19.0
- PyPDF2==2.0.1
- python-docx==0.8.11
- pytesseract==0.3.8
- Pillow==8.3.1
- requests==2.26.0
- groq==0.3.0

System Requirements:
- Tesseract OCR (for PDF processing)
- At least 4GB RAM
- 2GB free disk space

## Usage

1. **Upload Documents**
   - Click the upload area or drag and drop files
   - Supported formats: PDF, DOCX, TXT
   - Maximum file size: 10MB per file
   - Wait for the upload and processing to complete

2. **Ask Questions**
   - Once files are uploaded, the query section will be enabled
   - Enter your question in the text area
   - Select a query mode (default, precise, or creative)
   - Click "Ask Question" to get your answer

3. **View Results**
   - The answer will be displayed with relevant sources
   - Sources are linked to specific parts of your documents
   - Error messages will be shown if something goes wrong

## Technical Details

- **Frontend**
  - Modern CSS with CSS variables for theming
  - Responsive design with mobile support
  - Smooth animations and transitions
  - Real-time status updates
  - Error handling and user feedback

- **Backend**
  - Python Flask server
  - Document processing pipeline
  - Natural language processing for question answering
  - Source extraction and attribution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 