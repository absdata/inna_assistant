from typing import List, Dict, Any, Optional
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from datetime import datetime
import re
import tiktoken

# Create logger for this module
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        logger.info("Initializing Document Processor service...")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Approximately 500 tokens
            chunk_overlap=400,  # Approximately 100 tokens
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        logger.info("Document Processor service initialized successfully")
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))
    
    def _extract_headers(self, text: str) -> List[Dict[str, Any]]:
        """Extract headers and their positions from text."""
        headers = []
        lines = text.split('\n')
        header_pattern = re.compile(r'^#+\s+.*$|^[A-Z][^a-z\n]{0,20}$|^[A-Z].*:$')
        
        current_pos = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if header_pattern.match(line):
                headers.append({
                    'text': line,
                    'position': current_pos,
                    'line_number': i
                })
            current_pos += len(line) + 1  # +1 for newline
        
        return headers
    
    def _create_metadata(self, 
                        source_id: str,
                        source_type: str,
                        title: Optional[str],
                        chunk_index: int,
                        total_chunks: int,
                        section_title: Optional[str] = None) -> Dict[str, Any]:
        """Create metadata dictionary for a chunk."""
        return {
            "source_id": source_id,
            "source_type": source_type,
            "title": title,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "section_title": section_title,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def process_document(self,
                             content: str,
                             source_id: str,
                             source_type: str = "file",
                             title: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process document content into chunks with metadata."""
        try:
            logger.info(f"Processing document: {title or source_id}")
            
            # Extract headers for structure
            headers = self._extract_headers(content)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            total_chunks = len(chunks)
            
            # Process each chunk with metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                # Find the nearest header before this chunk
                chunk_start_pos = content.find(chunk)
                section_title = None
                
                if headers:
                    # Find the last header before this chunk
                    for header in reversed(headers):
                        if header['position'] <= chunk_start_pos:
                            section_title = header['text']
                            break
                
                # Create chunk with metadata
                processed_chunk = {
                    "content": chunk,
                    "metadata": self._create_metadata(
                        source_id=source_id,
                        source_type=source_type,
                        title=title,
                        chunk_index=i,
                        total_chunks=total_chunks,
                        section_title=section_title
                    )
                }
                processed_chunks.append(processed_chunk)
            
            logger.info(f"Successfully processed document into {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            raise
    
    async def process_file(self,
                          file_path: str,
                          source_id: str,
                          title: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a file based on its type."""
        try:
            logger.info(f"Processing file: {file_path}")
            
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                content = "\n\n".join(page.page_content for page in pages)
            
            elif file_path.lower().endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                content = loader.load()[0].page_content
            
            elif file_path.lower().endswith('.txt'):
                loader = TextLoader(file_path)
                content = loader.load()[0].page_content
            
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            return await self.process_document(
                content=content,
                source_id=source_id,
                source_type="file",
                title=title or file_path
            )
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            raise

# Create singleton instance
document_processor = DocumentProcessor() 