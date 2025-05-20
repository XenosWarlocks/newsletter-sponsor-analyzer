from dataclasses import asdict
from pathlib import Path
import csv
import logging
import hashlib
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
print(dir(genai))

from config import SponsorContent, DynamicRateLimiter, ScraperConfig
from typing import Optional, Set, Iterator, Dict
from datetime import datetime, date
from collections import defaultdict
from abc import ABC, abstractmethod
import concurrent.futures

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

class ContentProcessor(ABC):
    """Abstract base class for content processors"""
    @abstractmethod
    def process(self, content: str, issue_number: int) -> Optional[SponsorContent]:
        pass

class GeminiProcessor(ContentProcessor):
    """Gemini AI-based content processor"""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key) # type: ignore[attr-defined]
        self.model = genai.GenerativeModel("gemini-2.0-flash") # type: ignore[attr-defined]

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
            
            # Fix for error 3: Handle None case for sponsorship_date
            sponsorship_date_str = data.get('sponsorship_date')
            sponsorship_date: Optional[date] = None
            sponsorship_date = None
            if sponsorship_date_str:
                try:
                    sponsorship_date = datetime.strptime(sponsorship_date_str, '%Y-%m-%d').date()
                except ValueError:
                    logger.error(f"Invalid date format in issue {issue_number}")
                    return None
            
            # Make sure sponsorship_date is not None before creating SponsorContent
            if sponsorship_date is None:
                logger.error(f"Missing sponsorship date for issue {issue_number}")
                return None

            return SponsorContent(
                issue_number=issue_number,
                company_name=data['company_name'],
                website=self._clean_website(data['website']),
                industry=data['industry'],
                sponsorship_date=sponsorship_date,
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
        if "frontendfoc.us/link" in website:
            # Strip tracking parameters and redirects
            try:
                # Extract actual website from tracking URL if possible
                parsed_url = website.split("frontendfoc.us/link/")[-1].split("/")[1]
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
            # Fix for error 4: Add None check before calling find
            sponsor_tag = table.find("span", class_="tag-sponsor")
            if sponsor_tag:
                company_name = self._extract_company_name(table)
                if company_name and company_name.lower() in self.processed_companies[issue_number]:
                    logger.info(f"Skipping duplicate sponsor {company_name} in issue {issue_number}")
                    continue

                if company_name:
                    self.processed_companies[issue_number].add(company_name.lower())
                yield str(table)

    def _extract_company_name(self, table: BeautifulSoup) -> Optional[str]:
        # Company name extraction logic
        # Fix for error 4: Improved company name extraction with proper None checks
        sponsor_tag = table.find("span", class_="tag-sponsor")
        if not sponsor_tag:
            return None
        
        parent_cell = sponsor_tag.find_parent("td")
        if not parent_cell:
            return None
        
        strong_tag = parent_cell.find("strong") or parent_cell.find("b")
        if not strong_tag:
            return None
        return strong_tag.text.strip()

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
    """Main scraper class with parallel processing and dynamic rate limiting"""
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.rate_limiter = DynamicRateLimiter()
        self.extractor = ContentExtractor()
        self.processor = GeminiProcessor(config.api_key)
        self.writer = CSVWriter(config.output_file)

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self.config.headers)
        return session

    def scrape_range(self, start: int, end: int):
        """Parallel scraping of newsletter issues"""
        # Ensure we scrape in descending order
        issue_range = list(range(start, end - 1, -1))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.rate_limits['max_workers']
        ) as executor:
            # Submit all tasks
            future_to_issue = {
                executor.submit(self._process_issue, issue_num): issue_num
                for issue_num in issue_range
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_issue):
                issue_num = future_to_issue[future]
                try:
                    future.result() # This will raise any exceptions that occurred
                except Exception as e:
                    logger.error(f"Failed to process issue {issue_num}: {str(e)}")
                    
            # Wait for all tasks to complete - this is redundant with as_completed but ensures clarity
            logger.info("Waiting for all processing tasks to complete...")
            concurrent.futures.wait(future_to_issue.keys())
            logger.info("All processing tasks completed")

    def _process_issue(self, issue_num: int):
        """Process a single newsletter issue with rate limiting"""
        logger.info(f"Processing issue {issue_num}")
        
        try:
            # Fetch the issue with exponential backoff
            response = self._fetch_with_retry(issue_num)
            if not response:
                logger.warning(f"Could not fetch issue {issue_num}, skipping")
                return

            # Process sponsor contents
            processed_count = 0
            sponsors_found = 0
            for content in self.extractor.extract_sponsor_content(response.text, issue_num):
                sponsors_found += 1
                try:
                    # Apply rate limiting before AI processing
                    self.rate_limiter.wait()
                    
                    if processed := self.processor.process(content, issue_num):
                        self.writer.append_content(processed)
                        processed_count += 1
                        logger.info(f"Successfully processed sponsor for issue {issue_num}")
                except Exception as e:
                    logger.error(f"Failed to process sponsor content in issue {issue_num}: {str(e)}")
            
            # Reset rate limiter if processing was successful
            if processed_count > 0:
                self.rate_limiter.reset()
            
            logger.info(f"Issue {issue_num} processing completed. Found {sponsors_found} sponsors, processed {processed_count}.")
            
        except Exception as e:
            logger.error(f"Unexpected error processing issue {issue_num}: {str(e)}")
            # Still return successfully to not break the batch process
            return

    def _fetch_with_retry(self, issue_num: int) -> Optional[requests.Response]:
        """Fetch issue with dynamic rate limiting and retries"""
        url = self.config.base_url.format(issue_num)
        session = self._create_session()
        
        for attempt in range(5):  # Maximum 5 attempts
            try:
                # Apply rate limiting before request
                self.rate_limiter.wait()
                
                response = session.get(
                    url, 
                    timeout=self.config.rate_limits['request_timeout']
                )
                response.raise_for_status()
                
                # Reset rate limiter on successful request
                self.rate_limiter.reset()
                return response
            
            except (requests.RequestException, requests.Timeout) as e:
                logger.warning(f"Attempt {attempt + 1} failed for issue {issue_num}: {str(e)}")
                # The rate limiter's wait will progressively increase delay
                self.rate_limiter.backoff()
        
        logger.error(f"Failed to fetch issue {issue_num} after maximum retries")
        return None

def main():
    config = ScraperConfig(
        api_key="*******************************",
        output_file=Path("processed_sponsors.csv")
    )
    
    scraper = NewsletterScraper(config)
    
    try:
        logger.info("Starting newsletter scraping batch process")
        scraper.scrape_range(start=733, end=633)
        logger.info("Scraping process completed successfully")
    except Exception as e:
        logger.critical(f"Batch process failed with error: {str(e)}")
    finally:
        logger.info("Batch process execution finished")

if __name__ == "__main__":
    main()

