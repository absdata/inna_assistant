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

class TelegramBotService:
    def __init__(self):
        self.application = Application.builder().token(config.telegram_token).build()
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up message handlers."""
        self.application.add_handler(
            MessageHandler(filters.TEXT | filters.Document.ALL, self._handle_message)
        )
    
    async def _extract_text_from_file(self, file: BinaryIO, file_name: str) -> Optional[str]:
        """Extract text content from supported file types."""
        try:
            if file_name.lower().endswith('.pdf'):
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            
            elif file_name.lower().endswith('.docx'):
                doc = Document(file)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            
            elif file_name.lower().endswith('.txt'):
                return file.read().decode('utf-8')
            
            return None
        except Exception as e:
            print(f"Error extracting text from file: {e}")
            return None
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages."""
        if not update.message:
            return
        
        chat_id = update.message.chat_id
        message_id = update.message.message_id
        user_id = update.message.from_user.id
        username = update.message.from_user.username
        text = update.message.text or ""
        file_url = None
        file_content = None
        
        # Handle file attachments
        if update.message.document:
            file = await update.message.document.get_file()
            file_url = file.file_path
            
            # Download and process file content
            with tempfile.NamedTemporaryFile() as temp_file:
                await file.download_to_memory(temp_file)
                temp_file.seek(0)
                file_content = await self._extract_text_from_file(
                    temp_file,
                    update.message.document.file_name
                )
        
        # Save message to database
        saved_message = await db_service.save_message(
            chat_id=chat_id,
            message_id=message_id,
            user_id=user_id,
            username=username,
            text=text,
            file_url=file_url,
            file_content=file_content
        )
        
        # Generate and save embedding
        if text or file_content:
            content_to_embed = (text + "\n" + (file_content or "")).strip()
            embedding = await openai_service.get_embedding(content_to_embed)
            
            await db_service.save_embedding(
                message_id=saved_message["id"],
                chat_id=chat_id,
                text=content_to_embed,
                embedding=embedding
            )
        
        # Check if message is addressed to Inna
        should_respond = any(
            text.lower().startswith(trigger)
            for trigger in config.agent_name_triggers
        )
        
        if should_respond:
            # Initialize agent state
            state = AgentState(
                current_message=saved_message,
                chat_id=chat_id
            )
            
            # Run the agent workflow
            try:
                result = await agent.ainvoke(state)
                if result.response:
                    await update.message.reply_text(result.response)
            except Exception as e:
                error_message = "I apologize, but I encountered an error while processing your request."
                await update.message.reply_text(error_message)
                print(f"Error in agent workflow: {e}")
    
    async def start(self):
        """Start the Telegram bot."""
        await self.application.initialize()
        await self.application.start()
        await self.application.run_polling()
    
    async def stop(self):
        """Stop the Telegram bot."""
        await self.application.stop()

bot_service = TelegramBotService()