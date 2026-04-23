import sys
import os
import json
import tempfile
import cgi
from http.server import BaseHTTPRequestHandler

# Add the backend directory to the Python path before importing
backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from models import UploadResponse, DocumentData, DocumentPage
from pdf_processor import PDFProcessor

# Vercel payload limit is 4.5MB for the entire request
MAX_PAYLOAD_SIZE = 4.5 * 1024 * 1024  # 4.5MB in bytes
MAX_FILE_SIZE = 4.5 * 1024 * 1024  # 4.5MB per individual file
MAX_TOTAL_FILES = 100  # Keep original limit
CHUNK_SIZE = 3.5 * 1024 * 1024  # Process in 3.5MB chunks to stay under limit


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Check if this is a chunked upload request
            chunk_info = self._parse_chunk_info()

            if chunk_info:
                return self._handle_chunked_upload(chunk_info)
            else:
                return self._handle_regular_upload()

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "error": f"Internal server error: {str(e)}",
                        "suggestion": "Please try again with fewer or smaller files",
                    }
                ).encode()
            )

    def _parse_chunk_info(self):
        """Parse chunk information from headers"""
        chunk_index = self.headers.get("X-Chunk-Index")
        total_chunks = self.headers.get("X-Total-Chunks")
        upload_id = self.headers.get("X-Upload-ID")

        if chunk_index and total_chunks and upload_id:
            return {
                "chunk_index": int(chunk_index),
                "total_chunks": int(total_chunks),
                "upload_id": upload_id,
            }
        return None

    def _handle_chunked_upload(self, chunk_info):
        """Handle individual chunk of a larger upload"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Chunk-Index, X-Total-Chunks, X-Upload-ID",
        )
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        # Process this chunk - for simplicity, treat each chunk as a complete upload
        # In a production system, you'd store chunks and combine them
        try:
            chunk_result = self._process_files()

            if isinstance(chunk_result, dict) and "error" in chunk_result:
                # If there's an error, return it immediately
                self.wfile.write(json.dumps(chunk_result).encode())
                return

            # For chunked uploads, return the documents directly instead of chunk progress
            # This simulates processing each chunk immediately
            self.wfile.write(json.dumps(chunk_result).encode())

        except Exception as e:
            error_response = {"error": f"Chunk processing failed: {str(e)}"}
            self.wfile.write(json.dumps(error_response).encode())

    def _handle_regular_upload(self):
        """Handle regular upload, potentially splitting into chunks if needed"""
        # Check content length first
        content_length = int(self.headers.get("content-length", 0))

        # Parse multipart form data
        content_type = self.headers.get("content-type", "")
        if not content_type.startswith("multipart/form-data"):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"error": "Content-Type must be multipart/form-data"}
                ).encode()
            )
            return

        # If payload is too large, suggest chunked upload
        if content_length > MAX_PAYLOAD_SIZE:
            self.send_response(413)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            error_response = {
                "error": "Request too large for single upload",
                "details": f"Upload size ({content_length / 1024 / 1024:.1f}MB) exceeds 4.5MB limit",
                "solution": "chunked_upload_required",
                "chunked_upload_info": {
                    "max_chunk_size_mb": 3.5,
                    "suggested_chunks": max(1, int(content_length / CHUNK_SIZE) + 1),
                    "instructions": "Split your upload into smaller batches and use chunked upload headers",
                },
                "limits": {
                    "max_payload_size_mb": 4.5,
                    "max_files_total": MAX_TOTAL_FILES,
                    "max_file_size_mb": 4.5,
                },
            }
            self.wfile.write(json.dumps(error_response).encode())
            return

        # Set CORS headers for successful request
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Chunk-Index, X-Total-Chunks, X-Upload-ID",
        )
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        # Process the upload normally
        result = self._process_files()
        self.wfile.write(json.dumps(result).encode())

    def _process_files(self):
        """Process uploaded files"""
        # Parse the form data
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers["content-type"],
            },
        )

        # Get description from form
        description = form.getvalue("description", "")

        # Get uploaded files
        files = []
        if "files" in form:
            if isinstance(form["files"], list):
                files = form["files"]
            else:
                files = [form["files"]]

        # Validate number of files
        if len(files) > MAX_TOTAL_FILES:
            return {
                "error": f"Too many files. Maximum {MAX_TOTAL_FILES} files allowed",
                "received": len(files),
                "suggestion": "Please upload files in smaller batches using chunked upload",
            }

        # Initialize PDF processor
        pdf_processor = PDFProcessor()

        # Process files and extract text
        documents = []
        total_processed_size = 0

        for i, file_item in enumerate(files):
            if not hasattr(file_item, "filename") or not file_item.filename:
                continue

            if not file_item.filename.endswith(".pdf"):
                return {
                    "error": f"Only PDF files are allowed. Found: {file_item.filename}"
                }

            # Read file data
            file_data = file_item.file.read()
            file_size = len(file_data)
            total_processed_size += file_size

            # Check individual file size
            if file_size > MAX_FILE_SIZE:
                return {
                    "error": f"File too large: {file_item.filename}",
                    "file_size_mb": round(file_size / 1024 / 1024, 1),
                    "max_size_mb": round(MAX_FILE_SIZE / 1024 / 1024, 1),
                    "suggestion": "Please reduce file size to under 4.5MB",
                }

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_data)
                temp_file_path = temp_file.name

            try:
                # Extract text from PDF
                pages_data = pdf_processor.extract_pages(temp_file_path)

                # Convert to DocumentPage objects
                pages = [
                    DocumentPage(page_number=page["page_number"], text=page["text"])
                    for page in pages_data
                ]

                documents.append(
                    DocumentData(
                        id=i + 1,
                        filename=file_item.filename,
                        pages=pages,
                        total_pages=len(pages),
                    )
                )
            except Exception as e:
                error_msg = f"Error processing {file_item.filename}: {str(e)}"
                print(f"PDF processing error: {error_msg}")
                return {
                    "error": error_msg,
                    "suggestion": "Please ensure the PDF is not corrupted and try again",
                }
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

        # Create response
        response = UploadResponse(
            documents=documents,
            message=f"Successfully processed {len(files)} documents ({total_processed_size / 1024 / 1024:.1f}MB total)",
        )

        return response.model_dump()

    def _process_chunk(self, chunk_info):
        """Process a single chunk of files"""
        # Process the chunk and return the result
        return self._process_files()

    def _finalize_chunked_upload(self, upload_id):
        """Finalize a chunked upload by combining all chunks"""
        # In a real implementation, you'd combine all stored chunks
        # For now, return a placeholder response
        return {
            "status": "upload_complete",
            "upload_id": upload_id,
            "message": "Chunked upload completed successfully",
            "note": "Full chunked upload implementation requires persistent storage",
        }

    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Chunk-Index, X-Total-Chunks, X-Upload-ID",
        )
        self.end_headers()

    def do_GET(self):
        # Add GET method for testing and showing limits
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        response_data = {
            "message": "Upload endpoint",
            "method": "POST",
            "description": "Upload PDF files with automatic chunking for large uploads",
            "limits": {
                "max_payload_size_mb": 4.5,
                "max_files_total": MAX_TOTAL_FILES,
                "max_file_size_mb": round(MAX_FILE_SIZE / 1024 / 1024, 1),
                "supported_formats": ["PDF"],
            },
            "chunked_upload": {
                "enabled": True,
                "chunk_size_mb": round(CHUNK_SIZE / 1024 / 1024, 1),
                "headers_required": ["X-Chunk-Index", "X-Total-Chunks", "X-Upload-ID"],
            },
            "recommendations": [
                "For uploads over 4.5MB, use chunked upload automatically",
                "Individual PDFs can be up to 4.5MB each",
                "Up to 100 files total supported",
            ],
        }

        self.wfile.write(json.dumps(response_data).encode())
