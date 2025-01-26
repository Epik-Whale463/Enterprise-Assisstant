# Enterprise AI Assistant

A Flask-based web application that provides advanced document processing, summarization, and conversational AI capabilities for enterprise use.

## Features

### Document Processing
- Support for multiple file formats (PDF, DOCX, RTF, TXT)
- Smart text extraction and preprocessing
- Secure file handling with automatic cleanup

### AI Capabilities
- **Document Summarization**
  - Extractive summarization using LSA
  - Abstractive summarization using BART
  - Configurable summary lengths (short, medium, long)

- **Keyword Extraction**
  - Technical terms identification
  - Key phrases extraction
  - Named entity recognition using spaCy

- **Conversational AI**
  - RAG (Retrieval Augmented Generation) implementation
  - Context-aware responses using Google's Gemini Pro
  - Persistent conversation history

### System Features
- Real-time performance metrics
- Health monitoring dashboard
- Secure session management
- Modular architecture for easy extension

## Technical Stack

- **Backend Framework**: Flask
- **AI/ML Libraries**:
  - Transformers (BART)
  - spaCy
  - Langchain
  - HuggingFace
  - Google Generative AI (Gemini Pro)
- **Vector Database**: Qdrant
- **Document Processing**: PyPDF2, python-docx, textract

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

## Configuration

Set the following environment variables:
```bash
GOOGLE_API_KEY=[your-google-api-key]
QDRANT_URL=[your-qdrant-url]
QDRANT_API_KEY=[your-qdrant-api-key]
HUGGINGFACEHUB_API_TOKEN=[your-huggingface-token]
```

## Usage

1. Start the server:
```bash
python app.py
```

2. Access the application at `http://localhost:8080`

3. Navigate to different sections:
   - Document Processing
   - HR Policies
   - IT Support
   - Company Events
   - Content Moderation
   - Security Settings

## API Endpoints

### Document Processing
- `POST /upload-document`: Upload and process documents
- `POST /generate-summary`: Generate document summaries
- `POST /extract-keywords`: Extract keywords and entities
- `POST /converse`: Interact with the conversational AI

### System
- `GET /health`: System health check
- `GET /dashboard`: Performance metrics and statistics

## Security Features

- Secure file upload handling
- Session-based user management
- Input validation and sanitization
- Automatic file cleanup
- Rate limiting (configurable)

## Performance Monitoring

The dashboard provides real-time metrics:
- Average summary generation time
- Average keyword extraction time
- Total requests processed
- Service health status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]