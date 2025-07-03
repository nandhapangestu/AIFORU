# AI For U - Document Q&A System

## Overview

This is a Streamlit-based document Q&A application that integrates Google Drive for file management with AI-powered question answering capabilities. The system allows users to upload documents to Google Drive, process them using LangChain and OpenAI, and then ask questions about the content. The application uses vector embeddings and retrieval-augmented generation (RAG) to provide accurate answers based on document content.

## System Architecture

The application follows a simple web-based architecture:

- **Frontend**: Streamlit web interface for user interactions and chat-based Q&A
- **Backend**: Python-based processing using LangChain for document analysis and OpenAI for AI responses
- **Storage**: Google Drive for document storage and management
- **AI Processing**: OpenAI GPT-4o model for question answering
- **Vector Database**: FAISS for document embeddings and similarity search

## Key Components

### 1. Document Management
- **Google Drive Integration**: Uses Google Drive API v3 with service account authentication
- **File Operations**: Upload, download, and list files from a specific Google Drive folder (ID: 1hkN3DA67cpUKiQaR23BUXc3-k_mVb5wr)
- **Supported Formats**: PDF, text files, and DOCX documents through specialized loaders (PyPDFLoader, TextLoader, Docx2txtLoader)

### 2. AI Processing Pipeline
- **Document Loading**: Multiple loaders for different file types handle various document formats
- **Text Processing**: CharacterTextSplitter breaks documents into manageable chunks for processing
- **Embeddings**: OpenAI embeddings convert text chunks to vector representations
- **Vector Store**: FAISS provides efficient similarity search and retrieval capabilities
- **Question Answering**: RetrievalQA chain using ChatOpenAI (GPT-4o) generates responses based on relevant document sections

### 3. Authentication & Security
- **Service Account**: Google service account (replit-drive-agent@project-ai-464108.iam.gserviceaccount.com) enables server-to-server authentication
- **API Keys**: OpenAI API key stored as environment variable with fallback to "default_key"
- **Scoped Access**: Limited Google Drive API access scope for security

### 4. User Interface
- **Streamlit App**: Wide layout with sidebar for document selection and processing
- **Chat Interface**: Text input for questions with chat history display
- **Document Browser**: Sidebar shows available documents and processing status
- **How-it-works Guide**: Educational content explaining the RAG process

## Data Flow

1. **Document Upload**: Users upload documents through Streamlit interface to Google Drive
2. **Document Processing**: System downloads files and processes them using appropriate LangChain loaders
3. **Text Chunking**: Documents are split into smaller chunks for better AI processing
4. **Embedding Generation**: Text chunks are converted to embeddings using OpenAI API
5. **Vector Storage**: Embeddings are stored in FAISS vector database for fast retrieval
6. **Query Processing**: User questions are embedded and matched against document vectors
7. **Answer Generation**: Most relevant document chunks are retrieved and used by GPT-4o to generate contextual answers
8. **Response Display**: AI responses are shown with source attribution in the chat interface

## External Dependencies

### Google Cloud Platform
- **Google Drive API**: For file storage and management with service account authentication
- **Project ID**: project-ai-464108

### OpenAI Services
- **OpenAI API**: For embeddings generation and GPT-4o chat completions
- **Model**: GPT-4o (released May 13, 2024) - do not change unless explicitly requested

### Python Libraries
- **Streamlit**: Web application framework
- **LangChain**: Document processing and RAG pipeline
- **FAISS**: Vector similarity search
- **Google API Client**: Google Drive integration
- **PyPDF/Docx2txt**: Document parsing

## Deployment Strategy

The application is designed for Replit deployment with the following considerations:

- **Environment Variables**: OPENAI_API_KEY must be set in Replit secrets
- **File Structure**: Service account credentials stored in `credentials/service_account.json`
- **Dependencies**: All requirements listed in requirements.txt for easy installation
- **Configuration**: Hardcoded folder ID and service account file path for simple setup

## Changelog

- July 03, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.