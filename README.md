# Knowledge Base RAG System - Setup Guide

This document provides instructions for setting up and running the Knowledge Base RAG System.

## System Requirements

- Python 3.12.7
- Node.js 20.19.5
- npm 10.8.2
- Windows 10/11 (win32 x64)

## Directory Structure

- `D:\Knowledge base new\Document for test` - Directory for documents to be processed
- `D:\Knowledge base new\qdrant` - Directory containing Qdrant CLI executable

## Setup Instructions

### 1. Install Python Dependencies

Open a command prompt and navigate to the project directory:

```
cd "D:\Knowledge base new"
```

Install the required Python packages:

```
pip install -r requirements.txt
```

### 2. Install Angular Frontend Dependencies

Navigate to the frontend directory and install dependencies:

```
cd "D:\Knowledge base new\frontend"
npm install
```

## Running the System

### Option 1: Start Everything at Once

Run the combined startup script:

```
cd "D:\Knowledge base new"
start_system.bat
```

### Option 2: Start Backend and Frontend Separately

#### Start the Backend:

```
cd "D:\Knowledge base new"
python start_backend.py
```

#### Start the Frontend:

```
cd "D:\Knowledge base new\frontend"
npm start
```

## Accessing the Application

Once both backend and frontend are running:

1. Open your web browser
2. Navigate to: http://localhost:4200

## Testing the System

1. Place some test documents in `D:\Knowledge base new\Document for test`
2. The system will automatically process these documents
3. Use the web interface to query your knowledge base
4. The system will return answers based on the content of your documents

## System Architecture

- **Backend (FastAPI)**: Handles document processing, embedding generation, and query processing
- **Frontend (Angular)**: Provides user interface for querying and viewing document list
- **Qdrant**: Vector database for storing and retrieving document embeddings
- **MPT-7B-Instruct**: Local LLM for generating answers based on retrieved context
- **BAAI Large**: Embedding model for converting text to vector representations

## Troubleshooting

- If the backend fails to start, check that Python and all dependencies are installed correctly
- If the frontend fails to start, check that Node.js and npm are installed correctly
- If documents aren't being processed, check that the file watcher is running and monitoring the correct directory
- If queries return no results, check that documents have been properly processed and stored in Qdrant

## Additional Information

- The system is fully local and does not require internet access after initial setup
- All document processing happens automatically when files are added, modified, or deleted
- The frontend automatically refreshes the document list every 10 seconds