# Telegram bot logic

import asyncio
from typing import Optional, BinaryIO
import tempfile
from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters
)
from config.config import config
from services.database import db_service
from services.azure_openai import openai_service
from agent.graph import agent, AgentState
import PyPDF2
from docx import Document
import logging

# Create logger for this module
logger = logging.getLogger(__name__)

class TelegramBotService:
    def __init__(self):
        logger.info("Initializing Telegram bot service...")
        self.application = Application.builder().token(config.telegram_token).build()
        self._setup_handlers()
        self._running = False
        logger.info("Telegram bot service initialized successfully")
    
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
                    if file_content:
                        logger.info("File content extracted successfully")
                    else:
                        logger.warning("Failed to extract file content")
            
            # Save message to database
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
            logger.info(f"Message saved to database with ID: {saved_message['id']}")
            
            # Generate and save embedding
            if text or file_content:
                logger.debug("Generating embedding for message content...")
                content_to_embed = (text + "\n" + (file_content or "")).strip()
                try:
                    embedding = await openai_service.get_embedding(content_to_embed)
                    
                    await db_service.save_embedding(
                        message_id=saved_message["id"],
                        chat_id=chat_id,
                        text=content_to_embed,
                        embedding=embedding
                    )
                    logger.info("Message embedding saved to database")
                except Exception as embed_error:
                    logger.error(f"Error generating or saving embedding: {str(embed_error)}", exc_info=True)
                    # Continue processing even if embedding fails
            
            # Check if message is addressed to Inna
            should_respond = any(
                text.lower().startswith(trigger)
                for trigger in config.agent_name_triggers
            )
            
            if should_respond:
                logger.info("Message addressed to Inna, preparing response...")
                # Initialize agent state as dictionary
                state = {
                    "current_message": saved_message,
                    "chat_id": chat_id,
                    "messages": [],
                    "context": [],
                    "plan": "",
                    "response": ""
                }
                
                # Run the agent workflow
                try:
                    logger.debug("Running agent workflow...")
                    result = await agent.ainvoke(state)
                    if result and isinstance(result, dict) and result.get("response"):
                        logger.info("Sending response to user...")
                        for attempt in range(3):  # Try up to 3 times
                            try:
                                await update.message.reply_text(result["response"])
                                logger.debug(f"Response sent: {result['response'][:100]}{'...' if len(result['response']) > 100 else ''}")
                                break
                            except Exception as reply_error:
                                if attempt == 2:  # Last attempt
                                    logger.error(f"Failed to send response after 3 attempts: {str(reply_error)}", exc_info=True)
                                    raise
                                else:
                                    logger.warning(f"Failed to send response (attempt {attempt + 1}), retrying...")
                                    await asyncio.sleep(1)  # Wait before retry
                    else:
                        logger.warning("Agent workflow completed but no response generated")
                except Exception as e:
                    logger.error(f"Error in agent workflow: {str(e)}", exc_info=True)
                    for attempt in range(3):  # Try up to 3 times
                        try:
                            error_message = "I apologize, but I encountered an error while processing your request."
                            await update.message.reply_text(error_message)
                            break
                        except Exception as reply_error:
                            if attempt == 2:  # Last attempt
                                logger.error(f"Failed to send error message after 3 attempts: {str(reply_error)}", exc_info=True)
                            else:
                                logger.warning(f"Failed to send error message (attempt {attempt + 1}), retrying...")
                                await asyncio.sleep(1)  # Wait before retry
            else:
                logger.debug("Message not addressed to Inna, ignoring")
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            for attempt in range(3):  # Try up to 3 times
                try:
                    await update.message.reply_text(
                        "I encountered an error while processing your message. Please try again later."
                    )
                    break
                except Exception as reply_error:
                    if attempt == 2:  # Last attempt
                        logger.error(f"Failed to send error message after 3 attempts: {str(reply_error)}", exc_info=True)
                    else:
                        logger.warning(f"Failed to send error message (attempt {attempt + 1}), retrying...")
                        await asyncio.sleep(1)  # Wait before retry
    
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
            
            # Use update polling instead of run_polling
            async with self.application:
                await self.application.updater.start_polling()
                logger.info("Telegram bot is now polling for updates")
                
                # Keep the polling running
                while self._running:
                    await asyncio.sleep(1)
                    
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
            if self.application.updater and self.application.updater.running:
                await self.application.updater.stop()
            await self.application.stop()
            logger.info("Telegram bot stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {str(e)}", exc_info=True)
            raise

bot_service = TelegramBotService()