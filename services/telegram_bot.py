# Telegram bot logic

import asyncio
from typing import Optional, BinaryIO
import tempfile
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters
)
from config.config import config
from services.database import db_service
from services.azure_openai import openai_service
from agent.graph import agent
import PyPDF2
from docx import Document
import logging
import re

# Create logger for this module
logger = logging.getLogger(__name__)

class TelegramBotService:
    def __init__(self):
        logger.info("Initializing Telegram bot service...")
        self.application = Application.builder().token(config.telegram_token).build()
        self._setup_handlers()
        self._running = False
        self._shutdown_event = asyncio.Event()
        logger.info("Telegram bot service initialized successfully")
    
    def _escape_markdown_v2(self, text: str) -> str:
        """Escape special characters for Telegram's MarkdownV2 format."""
        if not text:
            return ""

        # First, escape the backslash itself
        text = text.replace('\\', '\\\\')
        
        # Characters that need escaping in MarkdownV2, in order of precedence
        special_chars = [
            # First escape characters that might be part of markdown syntax
            '_', '*', '[', ']', '(', ')', '~', '`', '>', '#',
            # Then escape general punctuation
            '+', '-', '=', '|', '{', '}', '.', '!',
        ]
        
        # Escape each special character with a backslash
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        
        # Handle newlines - replace \n with actual newline
        text = text.replace('\\n', '\n')
        
        # Handle common markdown patterns to ensure they work
        # This prevents double-escaping of already properly formatted markdown
        markdown_patterns = [
            (r'\\\*(.*?)\\\*', r'*\1*'),  # Bold
            (r'\\_(.*?)\\_', r'_\1_'),    # Italic
            (r'\\\`(.*?)\\\`', r'`\1`'),  # Code
            (r'\\\[(.*?)\\\]\\\((.*?)\\\)', r'[\1](\2)')  # Links
        ]
        
        for pattern, replacement in markdown_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _setup_handlers(self):
        """Set up message handlers."""
        logger.debug("Setting up message handlers...")
        self.application.add_handler(
            MessageHandler(filters.TEXT | filters.Document.ALL, self._handle_message)
        )
        logger.debug("Message handlers set up successfully")
    
    async def _extract_text_from_file(self, file: BinaryIO, file_name: str) -> Optional[str]:
        """Extract text content from supported file types."""
        logger.debug(f"Attempting to extract text from file: {file_name}")
        try:
            if file_name.lower().endswith('.pdf'):
                logger.debug("Processing PDF file...")
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                logger.debug(f"Successfully extracted {len(text)} characters from PDF")
                return text
            
            elif file_name.lower().endswith('.docx'):
                logger.debug("Processing DOCX file...")
                doc = Document(file)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                logger.debug(f"Successfully extracted {len(text)} characters from DOCX")
                return text
            
            elif file_name.lower().endswith('.txt'):
                logger.debug("Processing TXT file...")
                text = file.read().decode('utf-8')
                logger.debug(f"Successfully extracted {len(text)} characters from TXT")
                return text
            
            logger.warning(f"Unsupported file type: {file_name}")
            return None
        except Exception as e:
            logger.error(f"Error extracting text from file {file_name}: {str(e)}", exc_info=True)
            return None
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages."""
        if not update.message:
            logger.warning("Received update without message")
            return
        
        chat_id = update.message.chat_id
        message_id = update.message.message_id
        user_id = update.message.from_user.id
        username = update.message.from_user.username
        text = update.message.text or ""
        
        logger.info(f"Received message from {username} (ID: {user_id}) in chat {chat_id}")
        logger.debug(f"Message content: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        file_url = None
        file_content = None
        
        try:
            # Handle file attachments
            if update.message.document:
                file = await update.message.document.get_file()
                file_url = file.file_path
                logger.info(f"Processing attached file: {update.message.document.file_name}")
                
                # Download and process file content
                with tempfile.NamedTemporaryFile() as temp_file:
                    logger.debug("Downloading file to temporary location...")
                    await file.download_to_memory(temp_file)
                    temp_file.seek(0)
                    file_content = await self._extract_text_from_file(
                        temp_file,
                        update.message.document.file_name
                    )
            
            # Save basic message first without file content
            logger.debug("Saving message to database...")
            saved_message = await db_service.save_message(
                chat_id=chat_id,
                message_id=message_id,
                user_id=user_id,
                username=username,
                text=text,
                file_url=file_url,
                file_content=file_content
            )
            
            if not saved_message:
                logger.error("Failed to save message")
                return
            
            logger.info(f"Message saved to database with ID: {saved_message['id']}")
            
            # If we have file content, save it in chunks
            if file_content:
                logger.info("Saving file content in chunks...")
                saved_chunks = await db_service.save_file_chunks(
                    message_id=saved_message["id"],
                    content=file_content
                )
                
                if saved_chunks:
                    # Generate embeddings for each chunk
                    logger.debug("Generating embeddings for file content chunks...")
                    chunk_size = 100000  # Same as in save_file_chunk
                    for i in range(0, len(file_content), chunk_size):
                        chunk = file_content[i:i + chunk_size]
                        chunk_index = i // chunk_size
                        
                        # Create a unique identifier for this chunk's embedding
                        chunk_identifier = f"chunk_{chunk_index}"
                        content_to_embed = (text + "\n" + chunk).strip()
                        
                        try:
                            embedding = await openai_service.get_embedding(content_to_embed)
                            # Save embedding with chunk reference
                            await db_service.save_embedding(
                                message_id=saved_message["id"],
                                chat_id=chat_id,
                                text=f"{chunk_identifier}: {content_to_embed}",
                                embedding=embedding,
                                chunk_index=chunk_index  # Add chunk_index to identify which chunk this embedding belongs to
                            )
                            logger.info(f"Saved embedding for chunk {chunk_index + 1}")
                        except Exception as embed_error:
                            logger.error(f"Error processing chunk {chunk_index + 1}: {str(embed_error)}", exc_info=True)
                            continue
                else:
                    logger.error("Failed to save file chunks")
            else:
                # Generate embedding for text-only message
                if text:
                    logger.debug("Generating embedding for message text...")
                    try:
                        embedding = await openai_service.get_embedding(text)
                        await db_service.save_embedding(
                            message_id=saved_message["id"],
                            chat_id=chat_id,
                            text=text,
                            embedding=embedding
                        )
                        logger.info("Message embedding saved to database")
                    except Exception as embed_error:
                        logger.error(f"Error generating or saving embedding: {str(embed_error)}", exc_info=True)
            
            # Check if message is addressed to Inna
            should_respond = any(
                text.lower().startswith(trigger)
                for trigger in config.agent_name_triggers
            )
            
            if should_respond:
                logger.info("Message addressed to Inna, preparing response...")
                # Initialize agent state as dictionary
                state = {
                    "messages": [],
                    "context": [],
                    "current_message": saved_message,
                    "chat_id": chat_id,
                    "plan": "",
                    "response": "",
                    "formatted_context": ""
                }
                
                # Run the agent workflow
                try:
                    logger.debug("Running agent workflow...")
                    result = await agent.ainvoke(state)
                    if result and isinstance(result, dict) and result.get("response"):
                        logger.info("Sending response to user...")
                        try:
                            response_text = self._escape_markdown_v2(result["response"])
                            logger.debug(f"Original response: {result['response'][:100]}{'...' if len(result['response']) > 100 else ''}")
                            logger.debug(f"Escaped response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
                            
                            for attempt in range(3):  # Try up to 3 times
                                try:
                                    await update.message.reply_text(
                                        response_text,
                                        parse_mode=ParseMode.MARKDOWN_V2
                                    )
                                    logger.debug(f"Response sent successfully on attempt {attempt + 1}")
                                    break
                                except Exception as reply_error:
                                    if attempt == 2:  # Last attempt
                                        logger.error(
                                            f"Failed to send response after 3 attempts. Error: {str(reply_error)}\n"
                                            f"Response text: {response_text[:500]}{'...' if len(response_text) > 500 else ''}"
                                        )
                                        raise
                                    else:
                                        logger.warning(f"Failed to send response (attempt {attempt + 1}). Error: {str(reply_error)}")
                                        await asyncio.sleep(1)  # Wait before retry
                        except Exception as format_error:
                            logger.error(f"Error formatting response: {str(format_error)}", exc_info=True)
                            # Try to send without formatting as a fallback
                            await update.message.reply_text(
                                "I encountered an error with message formatting. Here's the unformatted response:\n\n" + 
                                result["response"]
                            )
                    else:
                        logger.warning("Agent workflow completed but no response generated")
                except Exception as e:
                    logger.error(f"Error in agent workflow: {str(e)}", exc_info=True)
                    raise
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)
            raise
    
    async def start(self):
        """Start the Telegram bot."""
        if self._running:
            logger.warning("Telegram bot is already running")
            return

        logger.info("Starting Telegram bot...")
        try:
            logger.debug("Initializing application...")
            await self.application.initialize()
            logger.debug("Starting application...")
            await self.application.start()
            logger.info("Starting message polling...")
            self._running = True
            self._shutdown_event.clear()
            
            # Use update polling instead of run_polling
            async with self.application:
                await self.application.updater.start_polling(
                    poll_interval=1.0,
                    timeout=30,
                    bootstrap_retries=-1,
                    read_timeout=30,
                    write_timeout=30
                )
                logger.info("Telegram bot is now polling for updates")
                
                # Keep the polling running until shutdown event is set
                await self._shutdown_event.wait()
                    
        except Exception as e:
            self._running = False
            logger.error(f"Error starting Telegram bot: {str(e)}", exc_info=True)
            raise
    
    async def stop(self):
        """Stop the Telegram bot."""
        if not self._running:
            logger.warning("Telegram bot is not running")
            return

        logger.info("Stopping Telegram bot...")
        try:
            self._running = False
            self._shutdown_event.set()
            
            # Stop the updater first
            if self.application.updater and self.application.updater.running:
                logger.debug("Stopping updater...")
                await self.application.updater.stop()
            
            # Stop the application
            logger.debug("Stopping application...")
            await self.application.stop()
            
            # Wait a moment to ensure all connections are closed
            await asyncio.sleep(0.5)
            
            logger.info("Telegram bot stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {str(e)}", exc_info=True)
            raise

bot_service = TelegramBotService()