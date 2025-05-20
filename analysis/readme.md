# Newsletter Sponsor Analysis Tool

A powerful Python tool for analyzing sponsor overlap between tech newsletters, specifically designed for JavaScript Weekly and Frontend Focus.


## Overview

This tool analyzes company sponsors across different tech newsletters to identify overlaps, unique sponsors, and industry distribution patterns. It uses a combination of exact matching, fuzzy matching, and AI-powered analysis to provide comprehensive insights into newsletter sponsorship patterns.

## Features

- **Multi-level Matching**: Identifies sponsor overlaps using:
  - Exact matching (by URL and company name)
  - Fuzzy matching with customizable threshold
  - AI-powered relationship detection (with Google Gemini API)
- **Data Cleaning**: Advanced preprocessing of company names and URLs for better matching
- **Rich Analytics**: Generates detailed statistics and visualizations
- **Comprehensive Reports**: Creates CSV exports and a detailed HTML report
- **Industry Analysis**: Analyzes company distribution by industry
- **AI-powered Categorization**: Optional advanced industry categorization using Gemini AI

## Requirements

- Python 3.7+
- Pandas
- NumPy
- Matplotlib
- FuzzyWuzzy
- Google Generative AI Python SDK (for AI analysis)
- Pathlib
- tqdm

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/XenosWarlocks/newsletter-sponsor-analyzer.git
   cd newsletter-sponsor-analyzer/analysis
   ```

2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib fuzzywuzzy python-Levenshtein pathlib tqdm google-generativeai
   ```

3. Optional: For Venn diagram visualization, install matplotlib-venn:
   ```bash
   pip install matplotlib-venn
   ```

## Usage

### Basic Usage

```bash
python newsletter_comparison.py --js your/file/path/to/js.xlsx --ff your/file/path/to/ff.xlsx --output ./results
```

### With AI Analysis

To enable AI-powered analysis using Google Gemini:

```bash
python newsletter_comparison.py --js your/file/path/to/js.xlsx --ff your/file/path/to/ff.xlsx --output ./results --gemini-key YOUR_GEMINI_API_KEY
```

### All Options

```bash
python newsletter_comparison.py --js your/file/path/to/js.xlsx --ff your/file/path/to/ff.xlsx --output ./results --threshold 85 --gemini-key YOUR_GEMINI_API_KEY --advanced-categorization
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--js` | Path to JavaScript Weekly Excel file | Required |
| `--ff` | Path to Frontend Focus Excel file | Required |
| `--output` | Output directory for results | `./results` |
| `--threshold` | Similarity threshold for fuzzy matching (0-100) | 85 |
| `--gemini-key` | Google Gemini API key for AI analysis | None |
| `--skip-ai` | Skip AI-powered analysis | False |
| `--advanced-categorization` | Use AI for advanced industry categorization | False |

## Input Data Format

The tool expects Excel files with the following columns:
- `Company Name`: Name of the sponsoring company
- `URL`: Website URL of the company
- `Issue #`: Newsletter issue number
- `Industry`: Industry category of the company

## Output Files

The tool generates the following outputs in the specified directory:

- **CSV Reports**:
  - `exact_matches.csv`: Companies matched exactly by URL or name
  - `fuzzy_matches.csv`: Companies matched by fuzzy string matching
  - `ai_matches.csv`: Companies identified as related by AI analysis
  - `js_weekly_unique.csv`: Companies unique to JS Weekly
  - `frontend_focus_unique.csv`: Companies unique to Frontend Focus
  - `js_weekly_categorized.csv`: Advanced industry categorization (if enabled)

- **Visualizations**:
  - `match_distribution.png`: Distribution of matches between newsletters
  - `industry_comparison.png`: Top industries by newsletter
  - `match_percentage_by_industry.png`: Match percentage by industry
  - `company_overlap_venn.png`: Venn diagram of company overlap

- **Reports**:
  - `enhanced_analytics_summary.txt`: Text summary of key statistics
  - `ai_company_insights.txt`: AI-generated insights about companies
  - `newsletter_analysis_report.html`: Comprehensive HTML report with all results

## How It Works

1. **Data Loading & Cleaning**:
   - Loads newsletter data from Excel files
   - Cleans and normalizes company names and URLs

2. **Matching Process**:
   - **Exact Matching**: Identifies identical companies by URL or name
   - **Fuzzy Matching**: Uses string similarity to match similar companies
   - **AI Analysis**: Uses Google Gemini to identify related companies based on semantic understanding

3. **Analysis & Visualization**:
   - Calculates match statistics and percentages
   - Analyzes industry distribution
   - Generates visualizations and reports

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FuzzyWuzzy for fuzzy string matching
- Google Generative AI for semantic analysis
- Matplotlib and Matplotlib-venn for visualizations
