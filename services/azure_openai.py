from openai import AzureOpenAI
from typing import List, Dict, Any
from config.config import config
import logging
import asyncio
from functools import partial
from utils.embedding_compressor import compressor

# Create logger for this module
logger = logging.getLogger(__name__)

class AzureOpenAIService:
    def __init__(self):
        logger.info("Initializing Azure OpenAI service...")
        self.client = AzureOpenAI(
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint
        )
        logger.info("Azure OpenAI service initialized successfully")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Azure OpenAI and compress it."""
        logger.debug(f"Getting embedding for text: {text[:100]}{'...' if len(text) > 100 else ''}")
        try:
            # Run the synchronous API call in a thread pool
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                partial(
                    self.client.embeddings.create,
                    model=config.azure_embedding_deployment,
                    input=text
                )
            )
            
            # Get the raw embedding
            raw_embedding = response.data[0].embedding
            
            # Compress the embedding to target dimensions
            compressed_embedding = compressor.compress(raw_embedding)
            
            logger.debug(f"Successfully generated and compressed embedding from {len(raw_embedding)} to {len(compressed_embedding)} dimensions")
            return compressed_embedding
        except Exception as e:
            logger.error(f"Error generating or compressing embedding: {str(e)}", exc_info=True)
            raise

    async def get_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Get completion from Azure OpenAI."""
        logger.debug(f"Getting completion for {len(messages)} messages")
        try:
            # Run the synchronous API call in a thread pool
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                partial(
                    self.client.chat.completions.create,
                    model=config.azure_deployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
            logger.debug("Successfully generated completion")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}", exc_info=True)
            raise

    def create_system_message(self, content: str) -> Dict[str, str]:
        """Create a system message for the chat completion."""
        return {"role": "system", "content": content}

    def create_user_message(self, content: str) -> Dict[str, str]:
        """Create a user message for the chat completion."""
        return {"role": "user", "content": content}

    def create_assistant_message(self, content: str) -> Dict[str, str]:
        """Create an assistant message for the chat completion."""
        return {"role": "assistant", "content": content}

openai_service = AzureOpenAIService() 