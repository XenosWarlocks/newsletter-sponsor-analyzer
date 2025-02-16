from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import time
import logging
import hashlib
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from typing import Optional, Set, Iterator, Dict
from datetime import datetime, date
from collections import defaultdict
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SponsorContent:
    """Data class for processed sponsor content"""
    issue_number: int
    company_name: str
    website: str
    industry: str
    sponsorship_date: date
    content_hash: str
    processed_at: datetime

class ScraperConfig:
    """Configuration management"""
    def __init__(self, api_key: str, output_file: Path):
        self.api_key = api_key
        self.output_file = output_file
        self.base_url = "https://javascriptweekly.com/issues/{}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.rate_limits = {
            'scrape_delay': 2,
            'ai_delay': 5,
            'ai_error_delay': 30,
            'max_retries': 5
        }

class ContentProcessor(ABC):
    """Abstract base class for content processors"""
    @abstractmethod
    def process(self, content: str, issue_number: int) -> Optional[SponsorContent]:
        pass

class GeminiProcessor(ContentProcessor):
    """Gemini AI-based content processor"""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def process(self, content: str, issue_number: int) -> Optional[SponsorContent]:
        prompt = self._create_prompt(content)
        try:
            response = self.model.generate_content(prompt)
            if not response.text:
                raise ValueError("Empty response from AI")
            
            # Parse and validate response
            data = self._parse_response(response.text, issue_number)
            if not data:
                return None

            return SponsorContent(
                issue_number=issue_number,
                company_name=data['company_name'],
                website=self._clean_website(data['website']),
                industry=data['industry'],
                sponsorship_date=datetime.strptime(data['sponsorship_date'], '%Y-%m-%d').date(),
                content_hash=hashlib.md5(content.encode('utf-8')).hexdigest(),
                processed_at=datetime.now()
            )
        except Exception as e:
            logger.error(f"AI processing failed for issue {issue_number}: {str(e)}")
            return None

    def _create_prompt(self, content: str) -> str:
        return f"""
        Analyze this newsletter sponsor content and extract the following information.
        Format your response as a JSON object with these exact keys:
        
        {{
            "company_name": "Name of the sponsoring company",
            "website": "Company's direct website (no tracking links)",
            "industry": "Company's primary industry/category",
            "sponsorship_date": "YYYY-MM-DD"
        }}

        Content to analyze:
        {content}

        Respond ONLY with the JSON object, no additional text.
        """

    def _parse_response(self, response: str, issue_number: int) -> Optional[Dict]:
        try:
            # Clean the response text
            response_text = response.strip()
            
            # Handle potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            
            # Parse JSON
            import json
            data = json.loads(response_text)
            
            # Validate required fields
            required_fields = {'company_name', 'website', 'industry', 'sponsorship_date'}
            if not all(field in data for field in required_fields):
                logger.error(f"Missing required fields in response for issue {issue_number}")
                return None
            
            # Validate date format
            try:
                datetime.strptime(data['sponsorship_date'], '%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid date format in response for issue {issue_number}")
                return None
                
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for issue {issue_number}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing response for issue {issue_number}: {str(e)}")
            return None

    def _clean_website(self, website: str) -> str:
        if "javascriptweekly.com/link" in website:
            # Strip tracking parameters and redirects
            try:
                # Extract actual website from tracking URL if possible
                parsed_url = website.split("javascriptweekly.com/link/")[-1].split("/")[1]
                return f"https://{parsed_url}"
            except:
                return website
        return website

class ContentExtractor:
    """Handles HTML content extraction"""
    def __init__(self):
        self.processed_companies: defaultdict[int, Set[str]] = defaultdict(set)

    def extract_sponsor_content(self, html: str, issue_number: int) -> Iterator[str]:
        soup = BeautifulSoup(html, "html.parser")
        for table in soup.find_all("table"):
            if table.find("span", class_="tag-sponsor"):
                company_name = self._extract_company_name(table)
                if company_name and company_name.lower() in self.processed_companies[issue_number]:
                    logger.info(f"Skipping duplicate sponsor {company_name} in issue {issue_number}")
                    continue
                
                if company_name:
                    self.processed_companies[issue_number].add(company_name.lower())
                yield str(table)

    def _extract_company_name(self, table: BeautifulSoup) -> Optional[str]:
        # Company name extraction logic
        sponsor_tag = table.find("span", class_="tag-sponsor")
        if sponsor_tag and sponsor_tag.find_parent("td"):
            parent_cell = sponsor_tag.find_parent("td")
            if strong_tag := (parent_cell.find("strong") or parent_cell.find("b")):
                return strong_tag.text.strip()
        return None

class CSVWriter:
    """Handles CSV file operations"""
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.fieldnames = [
            'issue_number', 'company_name', 'website', 'industry',
            'sponsorship_date', 'content_hash', 'processed_at'
        ]
        self._initialize_file()

    def _initialize_file(self):
        if not self.output_file.exists():
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def append_content(self, content: SponsorContent):
        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(asdict(content))

class NewsletterScraper:
    """Main scraper class"""
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.session = self._create_session()
        self.extractor = ContentExtractor()
        self.processor = GeminiProcessor(config.api_key)
        self.writer = CSVWriter(config.output_file)

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self.config.headers)
        return session

    def scrape_range(self, start: int, end: int):
        """Scrape a range of newsletter issues"""
        for issue_num in range(start, end - 1, -1):
            try:
                self._process_issue(issue_num)
            except Exception as e:
                logger.error(f"Failed to process issue {issue_num}: {str(e)}")
                continue

    def _process_issue(self, issue_num: int):
        logger.info(f"Processing issue {issue_num}")
        response = self._fetch_with_retry(issue_num)
        if not response:
            return

        for content in self.extractor.extract_sponsor_content(response.text, issue_num):
            try:
                if processed := self.processor.process(content, issue_num):
                    self.writer.append_content(processed)
                    logger.info(f"Successfully processed sponsor for issue {issue_num}")
                time.sleep(self.config.rate_limits['ai_delay'])
            except Exception as e:
                logger.error(f"Failed to process sponsor content in issue {issue_num}: {str(e)}")

    def _fetch_with_retry(self, issue_num: int) -> Optional[requests.Response]:
        url = self.config.base_url.format(issue_num)
        for attempt in range(self.config.rate_limits['max_retries']):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for issue {issue_num}: {str(e)}")
                if attempt < self.config.rate_limits['max_retries'] - 1:
                    time.sleep(self.config.rate_limits['scrape_delay'])
        return None

def main():
    config = ScraperConfig(
        api_key="Your_API_Key",
        output_file=Path("processed_sponsors.csv")
    )
    
    scraper = NewsletterScraper(config)
    scraper.scrape_range(start=723, end=700)

if __name__ == "__main__":
    main()
