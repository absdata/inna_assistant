from openai import AzureOpenAI
from typing import List, Dict, Any
from config.config import config

class AzureOpenAIService:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint
        )
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Azure OpenAI."""
        response = await self.client.embeddings.create(
            model=config.azure_embedding_deployment,
            input=text
        )
        return response.data[0].embedding

    async def get_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Get completion from Azure OpenAI."""
        response = await self.client.chat.completions.create(
            model=config.azure_deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

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