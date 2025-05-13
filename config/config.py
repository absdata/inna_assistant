import os
from dotenv import load_dotenv
from typing import Optional
import logging
from pathlib import Path

# Find and load .env file from project root
env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    """Central configuration class for Inna AI Assistant."""
    
    def __init__(self):
        # Configure logging first
        self._setup_logging()
        
        # Telegram Configuration
        self.telegram_token: str = self._get_env("TELEGRAM_BOT_TOKEN")
        
        # Azure OpenAI Configuration
        self.azure_api_key: str = self._get_env("AZURE_OPENAI_API_KEY")
        self.azure_endpoint: str = self._get_env("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version: str = self._get_env("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.azure_deployment: str = self._get_env("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.azure_embedding_deployment: str = self._get_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        
        # Supabase Configuration
        self.supabase_url: str = self._get_env("SUPABASE_URL")
        self.supabase_key: str = self._get_env("SUPABASE_KEY")
        
        # Application Configuration
        self.max_context_messages: int = int(self._get_env("MAX_CONTEXT_MESSAGES", "5"))
        self.embedding_dimension: int = int(self._get_env("EMBEDDING_DIMENSION", "2000"))
        self.debug_mode: bool = self._get_env("DEBUG_MODE", "false").lower() == "true"
        self.log_level: str = self._get_env("LOG_LEVEL", "INFO")

        # Google Doc Configuration
        self.google_service_account: str = self._get_env("GOOGLE_SERVICE_ACCOUNT")
        self.google_service_account_key: str = self._get_env("GOOGLE_SERVICE_ACCOUNT_KEY")
        
        # Agent Configuration
        self.agent_name_triggers = ["inna,", "ina,", "inna"]
    
    def _setup_logging(self) -> None:
        """Configure logging for the application."""
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()  # Output to console
            ]
        )
    
    @staticmethod
    def _get_env(key: str, default: Optional[str] = None) -> str:
        """
        Get environment variable with validation.
        
        Args:
            key: Environment variable name
            default: Optional default value
            
        Returns:
            str: Environment variable value
            
        Raises:
            ValueError: If required variable is not set
        """
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        return value

# Create singleton instance
config = Config()