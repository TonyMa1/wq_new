"""
Configuration management for the alpha generator.
Handles loading and validation of configuration from environment variables.
"""

import os
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import dotenv

# Try to load .env file if it exists
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class WorldQuantConfig:
    """WorldQuant API configuration."""
    
    username: str
    password: str
    max_retries: int = 3
    retry_delay: int = 5
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'WorldQuantConfig':
        """Load configuration from environment variables."""
        username = os.environ.get("WQ_USERNAME")
        password = os.environ.get("WQ_PASSWORD")
        
        if not username or not password:
            logger.error("WQ_USERNAME and WQ_PASSWORD must be set in environment variables")
            raise ValueError("Missing required WorldQuant credentials")
        
        return cls(
            username=username,
            password=password,
            max_retries=int(os.environ.get("WQ_MAX_RETRIES", "3")),
            retry_delay=int(os.environ.get("WQ_RETRY_DELAY", "5")),
            timeout=int(os.environ.get("WQ_TIMEOUT", "30"))
        )

@dataclass
class AIConfig:
    """AI service configuration."""
    
    api_key: str
    model: str = "google/gemini-2.5-pro-exp-03-25:free"
    max_retries: int = 3
    timeout: int = 90
    site_url: str = "http://localhost"
    site_name: str = "WorldQuantAlphaGen"
    
    @classmethod
    def from_env(cls) -> 'AIConfig':
        """Load configuration from environment variables."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not api_key:
            logger.error("OPENROUTER_API_KEY must be set in environment variables")
            raise ValueError("Missing required OpenRouter API key")
        
        return cls(
            api_key=api_key,
            model=os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-pro-exp-03-25:free"),
            max_retries=int(os.environ.get("AI_MAX_RETRIES", "3")),
            timeout=int(os.environ.get("AI_TIMEOUT", "90")),
            site_url=os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
            site_name=os.environ.get("OPENROUTER_SITE_NAME", "WorldQuantAlphaGen")
        )

@dataclass
class AppConfig:
    """Application configuration."""
    
    data_dir: str = "./data"
    log_level: str = "INFO"
    batch_size: int = 10
    concurrent_batches: int = 5
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables."""
        return cls(
            data_dir=os.environ.get("DATA_DIR", "./data"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            batch_size=int(os.environ.get("BATCH_SIZE", "10")),
            concurrent_batches=int(os.environ.get("CONCURRENT_BATCHES", "5"))
        )

class Config:
    """Main configuration class."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.wq = WorldQuantConfig.from_env()
        self.ai = AIConfig.from_env()
        self.app = AppConfig.from_env()
    
    @classmethod
    def load(cls) -> 'Config':
        """Load configuration from environment variables."""
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "wq": {
                "username": self.wq.username,
                "password": "********",  # Mask password for security
                "max_retries": self.wq.max_retries,
                "retry_delay": self.wq.retry_delay,
                "timeout": self.wq.timeout
            },
            "ai": {
                "api_key": "********",  # Mask API key for security
                "model": self.ai.model,
                "max_retries": self.ai.max_retries,
                "timeout": self.ai.timeout,
                "site_url": self.ai.site_url,
                "site_name": self.ai.site_name
            },
            "app": {
                "data_dir": self.app.data_dir,
                "log_level": self.app.log_level,
                "batch_size": self.app.batch_size,
                "concurrent_batches": self.app.concurrent_batches
            }
        }