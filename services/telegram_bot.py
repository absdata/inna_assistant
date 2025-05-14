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
from services.document_processor import document_processor
from agent.graph import agent, AgentState
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
            MessageHandler(
                filters.TEXT | filters.Document.ALL,
                self._handle_message
            )
        )
        logger.debug("Message handlers set up successfully")
    
    async def start(self):
        """Start the bot."""
        if not self._running:
            logger.info("Starting Telegram bot...")
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            self._running = True
            logger.info("Telegram bot is running")
    
    async def stop(self):
        """Stop the bot."""
        if self._running:
            logger.info("Stopping Telegram bot...")
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            self._running = False
            self._shutdown_event.set()
            logger.info("Telegram bot stopped")
    
    async def _extract_text_from_file(self, file: BinaryIO, file_name: str) -> Optional[str]:
        """Extract text content from supported file types."""
        logger.debug(f"Attempting to extract text from file: {file_name}")
        try:
            # Save file to temporary location
            with tempfile.NamedTemporaryFile(suffix=file_name[file_name.rfind('.'):]) as temp_file:
                temp_file.write(file.read())
                temp_file.flush()
                
                # Process file using document processor
                chunks = await document_processor.process_file(
                    file_path=temp_file.name,
                    source_id=file_name,
                    title=file_name
                )
                
                if chunks:
                    # Combine all chunks in order
                    return "\n\n".join(chunk["content"] for chunk in chunks)
                
                logger.warning(f"No content extracted from file: {file_name}")
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
            
            if not saved_message:
                logger.error("Failed to save message")
                await update.message.reply_text(
                    "Sorry, I couldn't process your message. Please try again.",
                    parse_mode=ParseMode.HTML
                )
                return
            
            # Initialize agent state
            initial_state = AgentState(
                messages=[],  # Previous messages if needed
                context=[],  # Will be populated by retrieve_context
                current_message={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "user_id": user_id,
                    "username": username,
                    "text": text,
                    "file_url": file_url,
                    "file_content": file_content
                },
                chat_id=chat_id
            )
            
            # Process message with agent
            result = await agent.ainvoke(initial_state)
            
            # Only respond if the agent actually processed the message
            if result and result.get("should_process"):
                if result.get("response"):
                    await update.message.reply_text(
                        result["response"],
                        parse_mode=ParseMode.HTML
                    )
                else:
                    await update.message.reply_text(
                        "I processed your message but couldn't generate a response. Please try again.",
                        parse_mode=ParseMode.HTML
                    )
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)
            await update.message.reply_text(
                "Sorry, I encountered an error while processing your message. Please try again.",
                parse_mode=ParseMode.HTML
            )

# Create singleton instance
telegram_bot = TelegramBotService()