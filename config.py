# config.py
from datetime import datetime, date
from dataclasses import dataclass, field
from pathlib import Path
import random
import time
import os
import logging
from typing import Dict, Any, Optional
import json

@dataclass
class SponsorContent:
    """Data class for processed sponsor content"""
    issue_number: int
    company_name: str
    website: str
    industry: str
    sponsorship_date: date
    content_hash: str
    processed_at: datetime = field(default_factory=datetime.now)

class DynamicRateLimiter:
    """Intelligent rate limiter with exponential backoff and improved configurability"""
    def __init__(
            self, 
            initial_delay: float = 1.0,
            max_delay: float = 60.0, 
            jitter_factor: float = 0.1,
            logger: Optional[logging.Logger] = None
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor
        self.current_delay = initial_delay
        self.logger = logger or logging.getLogger(__name__)

    def wait(self):
        """Wait with exponential backoff and added jitter"""
        try:
            # Add jitter to prevent synchronized retries
            jittered_delay = self.current_delay * (1 + random.uniform(-self.jitter_factor, self.jitter_factor))
            self.logger.debug(f"Rate limiter waiting for {jittered_delay:.2f} seconds")
            time.sleep(jittered_delay)

            # Increase delay exponentially, but cap at max_delay
            self.current_delay = min(self.current_delay * 2, self.max_delay)
        except Exception as e:
            self.logger.error(f"Error in rate limiter: {e}")
            raise

    def reset(self):
        """Reset the delay to initial value"""
        self.logger.debug("Rate limiter reset to initial delay")
        self.current_delay = self.initial_delay

class ScraperConfig:
    """Enhanced configuration management with environment and file-based configuration"""
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        output_file: Optional[Path] = None, 
        config_file: Optional[Path] = None
    ):
        # Priority: Explicit parameters > Environment Variables > Config File > Defaults
        self.config = self._load_configuration(config_file)
        
        # # API Key precedence
        # self.api_key = (
        #     api_key or 
        #     os.getenv('GEMINI_API_KEY') or 
        #     self.config.get('api_key')
        # )
        # API Key precedence
        api_key_from_env = os.getenv('GEMINI_API_KEY')
        api_key_from_config = self.config.get('api_key')
        
        self.api_key = api_key or api_key_from_env or api_key_from_config

        # Output file precedence
        output_file_from_env = os.getenv('OUTPUT_FILE')
        output_file_from_config = self.config.get('output_file')
        
        # # Output file precedence
        # self.output_file = (
        #     output_file or 
        #     Path(os.getenv('OUTPUT_FILE', 'processed_sponsors.csv')) or 
        #     Path(self.config.get('output_file', 'processed_sponsors.csv'))
        # )
        
        # # URL configuration
        # self.base_url = self.config.get(
        #     'base_url', 
        #     "https://javascriptweekly.com/issues/{}"
        # )

        if output_file:
            self.output_file = output_file
        elif output_file_from_env:
            self.output_file = Path(output_file_from_env)
        elif output_file_from_config:
            self.output_file = Path(output_file_from_config)
        else:
            self.output_file = Path('processed_sponsors.csv')

        # URL configuration
        self.base_url = self.config.get(
            'base_url', 
            "https://frontendfoc.us/issues/{}"
        )
        
        # Headers with optional customization
        self.headers = self.config.get('headers', {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        
        # Rate limits with flexible configuration
        self.rate_limits = {
            'max_workers': self.config.get('max_workers', 4),
            'ai_request_timeout': self.config.get('ai_request_timeout', 60),
            'request_timeout': self.config.get('request_timeout', 30),
        }
        
        # Validate critical configurations
        self._validate_config()

    def _load_configuration(self, config_file: Optional[Path] = None) -> Dict[str, Any]:
        """Load configuration from file with error handling"""
        default_config_paths = [
            Path('./scraper_config.json'),
            Path.home() / '.config' / 'newsletter_scraper' / 'config.json'
        ]
        
        config_file = config_file or next((path for path in default_config_paths if path.exists()), None)
        
        if config_file and config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logging.error(f"Error reading config file: {e}")
        
        return {}

    def _validate_config(self):
        """Validate critical configuration parameters"""
        if not self.api_key:
            raise ValueError("No API key provided. Set via parameter, env variable, or config file.")
        
        if not self.output_file:
            raise ValueError("No output file specified.")
