import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
import io
import os
import tempfile
import logging
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SERVICE_ACCOUNT_FILE = "credentials/service_account.json"
FOLDER_ID = "1hkN3DA67cpUKiQaR23BUXc3-k_mVb5wr"
SCOPES = ["https://www.googleapis.com/auth/drive"]

# Get OpenAI API key from environment with fallback
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
openai_api_key = os.getenv("OPENAI_API_KEY", "default_key")

# Streamlit page configuration
st.set_page_config(
    page_title="AI For U — Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Google Drive service
@st.cache_resource
def init_drive_service():
    """Initialize Google Drive service with service account credentials"""
    try:
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            st.error(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
            return None
        
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        service = build("drive", "v3", credentials=creds)
        logger.info("Google Drive service initialized successfully")
        return service
    except Exception as e:
        st.error(f"Failed to initialize Google Drive service: {str(e)}")
        logger.error(f"Drive service initialization error: {e}")
        return None

def list_drive_files(service, folder_id):
    """List files in the specified Google Drive folder"""
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, mimeType, size, modifiedTime)"
        ).execute()
        files = results.get("files", [])
        logger.info(f"Found {len(files)} files in Drive folder")
        return files
    except HttpError as e:
        st.error(f"Error listing Drive files: {str(e)}")
        logger.error(f"Drive listing error: {e}")
        return []

def download_file(service, file_id, dest_path):
    """Download a file from Google Drive"""
    try:
        request = service.files().get_media(fileId=file_id)
        with open(dest_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        logger.info(f"Downloaded file to {dest_path}")
        return True
    except HttpError as e:
        st.error(f"Error downloading file: {str(e)}")
        logger.error(f"Download error: {e}")
        return False

def upload_to_drive(service, file_path, folder_id, filename):
    """Upload a file to Google Drive"""
    try:
        file_metadata = {
            "name": filename,
            "parents": [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        uploaded_file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id"
        ).execute()
        file_id = uploaded_file.get("id")
        logger.info(f"Uploaded file with ID: {file_id}")
        return file_id
    except HttpError as e:
        st.error(f"Error uploading file: {str(e)}")
        logger.error(f"Upload error: {e}")
        return None

def process_document(file_path, file_name):
    """Process a document and create vector store"""
    try:
        # Choose appropriate loader based on file extension
        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_name.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            st.error("Unsupported file format")
            return None
        
        # Load and split documents
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            st.error("No text content found in the document")
            return None
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Create QA chain
        llm = ChatOpenAI(
            model_name="gpt-4o",  # Using the latest OpenAI model
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 4}
            ),
            return_source_documents=True
        )
        
        logger.info(f"Document processed successfully: {len(chunks)} chunks created")
        return qa_chain
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        logger.error(f"Document processing error: {e}")
        return None

# Main application
def main():
    st.title("🤖 AI For U — Document Q&A System")
    st.markdown("Upload documents to Google Drive and ask questions about their content using AI.")
    
    # Initialize Drive service
    drive_service = init_drive_service()
    if not drive_service:
        st.stop()
    
    # Check OpenAI API key
    if not openai_api_key or openai_api_key == "default_key":
        st.warning("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    
    # Sidebar for file management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload section
        st.subheader("📤 Upload New Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt"],
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_file is not None:
            if st.button("Upload to Drive", type="primary"):
                with st.spinner("Uploading file..."):
                    # Save uploaded file temporarily
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Upload to Google Drive
                    file_id = upload_to_drive(
                        drive_service, 
                        temp_path, 
                        FOLDER_ID, 
                        uploaded_file.name
                    )
                    
                    if file_id:
                        st.success(f"✅ File uploaded successfully! ID: {file_id}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to upload file")
                    
                    # Clean up temporary file
                    os.unlink(temp_path)
                    os.rmdir(temp_dir)
        
        st.markdown("---")
        
        # How it works section
        st.subheader("📘 How it works")
        st.markdown("""
        1. **Upload Documents**: Upload PDF, DOCX, or TXT files to Google Drive
        2. **Document Processing**: Files are split into chunks and converted to embeddings
        3. **Vector Search**: Questions are matched against document content using AI
        4. **AI Response**: GPT-4o analyzes relevant sections and provides answers
        5. **Source Attribution**: See which document parts were used for the answer
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Available Documents")
        
        # List files from Google Drive
        files = list_drive_files(drive_service, FOLDER_ID)
        supported_files = [
            f for f in files 
            if f['name'].lower().endswith(('.pdf', '.docx', '.txt'))
        ]
        
        if not supported_files:
            st.info("No documents found. Upload a document to get started.")
        else:
            # Create a selectbox for file selection
            file_options = {f['name']: f['id'] for f in supported_files}
            selected_file = st.selectbox(
                "Select a document to analyze:",
                options=list(file_options.keys()),
                key="file_selector"
            )
            
            if selected_file and st.button("🔍 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    # Download and process the selected file
                    temp_dir = tempfile.mkdtemp()
                    local_path = os.path.join(temp_dir, selected_file)
                    
                    if download_file(drive_service, file_options[selected_file], local_path):
                        qa_chain = process_document(local_path, selected_file)
                        
                        if qa_chain:
                            st.session_state.qa_chain = qa_chain
                            st.session_state.current_document = selected_file
                            st.success(f"✅ Document '{selected_file}' processed successfully!")
                        else:
                            st.error("❌ Failed to process document")
                    
                    # Clean up temporary files
                    if os.path.exists(local_path):
                        os.unlink(local_path)
                    os.rmdir(temp_dir)
    
    with col2:
        st.subheader("💬 Document Q&A")
        
        # Display current document
        if "current_document" in st.session_state:
            st.info(f"📄 Currently analyzing: **{st.session_state.current_document}**")
        else:
            st.info("📄 No document selected. Please select and process a document first.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("📚 Source Documents"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(source.page_content[:500] + "..." if len(source.page_content) > 500 else source.page_content)
                            st.markdown("---")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            if "qa_chain" not in st.session_state:
                st.error("Please select and process a document first!")
            else:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            result = st.session_state.qa_chain({"query": prompt})
                            response = result["result"]
                            sources = result.get("source_documents", [])
                            
                            st.markdown(response)
                            
                            # Display sources if available
                            if sources:
                                with st.expander("📚 Source Documents"):
                                    for i, source in enumerate(sources):
                                        st.markdown(f"**Source {i+1}:**")
                                        st.markdown(source.page_content[:500] + "..." if len(source.page_content) > 500 else source.page_content)
                                        st.markdown("---")
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "sources": sources
                            })
                            
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
        
        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
    main()
