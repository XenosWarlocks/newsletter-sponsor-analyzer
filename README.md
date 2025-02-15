# newsletter-sponsor-analyzer

A robust Python tool for extracting and analyzing sponsor content from JavaScript Weekly newsletters using AI-powered content processing.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This tool automatically scrapes and analyzes sponsor content from JavaScript Weekly newsletters, extracting key information about sponsoring companies using Google's Gemini AI.

### Key Features

- 🤖 AI-powered content analysis using Google's Gemini model
- 🔄 Robust rate limiting and error handling
- 📊 Structured data output in CSV format
- 🔍 Automatic duplicate sponsor detection
- 🧹 Clean website URL extraction from tracking links
- 📝 Comprehensive logging system

## Installation

```bash
# Clone the repository
git clone https://github.com/XenosWarlocks/newsletter-sponsor-analyzer
cd newsletter-sponsor-analyzer

# Install required packages
pip install -r requirements.txt
```

## Requirements

- Python 3.8 or higher
- Google AI API key (Gemini model access)
- Required Python packages:
  - beautifulsoup4
  - requests
  - google-generativeai
  - dataclasses (included in Python 3.7+)

## Usage

1. Set up your configuration:
   ```python
   config = ScraperConfig(
       api_key="your-google-ai-api-key",
       output_file=Path("processed_sponsors.csv")
   )
   ```

2. Initialize and run the scraper:
   ```python
   scraper = NewsletterScraper(config)
   scraper.scrape_range(start=723, end=700)  # Adjust range as needed
   ```

3. Check the output:
   - Results are saved in `processed_sponsors.csv`
   - Logs are written to `scraper.log`

## Output Format

The tool generates a CSV file with the following columns:
- `issue_number`: Newsletter issue identifier
- `company_name`: Name of the sponsoring company
- `website`: Company's clean website URL
- `industry`: Company's primary industry/category
- `sponsorship_date`: Date of sponsorship
- `content_hash`: MD5 hash of the original content
- `processed_at`: Timestamp of processing

## Architecture

The system is built with a modular architecture:

- `ScraperConfig`: Configuration management
- `ContentProcessor`: Abstract base class for content processing
- `GeminiProcessor`: AI-powered content analysis
- `ContentExtractor`: HTML parsing and content extraction
- `CSVWriter`: Data output handling
- `NewsletterScraper`: Main orchestration class

## Rate Limiting

The tool implements considerate rate limiting:
- 2-second delay between page scrapes
- 5-second delay between AI processing calls
- 30-second delay after AI errors
- Maximum of 5 retry attempts per issue

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Disclaimer

This tool is designed for research and analysis purposes. Please ensure you comply with JavaScript Weekly's terms of service and implement appropriate rate limiting when using this tool.
