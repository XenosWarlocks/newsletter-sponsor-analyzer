import pandas as pd
import numpy as np
from pathlib import Path
import re
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Optional
import argparse
import os
import google.generativeai as genai
from tqdm import tqdm

def configure_gemini(api_key: str):
    """Configure Gemini AI with API key."""
    genai.configure(api_key=api_key) # type: ignore[attr-defined]
    return genai.GenerativeModel('gemini-2.0-flash') # type: ignore[attr-defined]

def clean_url(url: str) -> str:
    """Clean URLs to make them comparable."""
    if not url or not isinstance(url, str):
        return ""
        
    # Normalize URLs - remove http/https, www, trailing slashes
    url = url.lower()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    url = re.sub(r'/$', '', url)
    
    # Handle common tracking parameters
    url = re.sub(r'\?.*', '', url)
    
    return url

def clean_company_name(name: str) -> str:
    """Clean company names to make them comparable."""
    if not name or not isinstance(name, str):
        return ""
        
    name = name.lower()
    # Remove common suffixes
    name = re.sub(r'\s+(inc\.?|llc|ltd\.?|gmbh|corp\.?|limited)$', '', name, flags=re.IGNORECASE)
    # Remove all non-alphanumeric characters
    name = re.sub(r'[^\w\s]', '', name)
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def load_data(js_weekly_path: str, frontend_focus_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from Excel files."""
    try:
        # Load data from Excel files
        js_weekly = pd.read_excel(js_weekly_path)
        frontend_focus = pd.read_excel(frontend_focus_path)
        
        # Basic data cleaning
        for df in [js_weekly, frontend_focus]:
            # Fill missing values
            df.fillna('', inplace=True)
            
            # Ensure company name and URL columns exist
            required_cols = ['Company Name', 'URL']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' missing in dataset")
        
        # Apply URL and company name cleaning
        js_weekly['Clean URL'] = js_weekly['URL'].astype(str).apply(clean_url)
        frontend_focus['Clean URL'] = frontend_focus['URL'].astype(str).apply(clean_url)
        js_weekly['Clean Company'] = js_weekly['Company Name'].astype(str).apply(clean_company_name)
        frontend_focus['Clean Company'] = frontend_focus['Company Name'].astype(str).apply(clean_company_name)
        
        # Add unique ID columns
        js_weekly['ID'] = js_weekly.index
        frontend_focus['ID'] = frontend_focus.index
        
        return js_weekly, frontend_focus
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def find_exact_matches(js_weekly: pd.DataFrame, frontend_focus: pd.DataFrame) -> pd.DataFrame:
    """Find exact matches based on URL or company name."""
    # Find URL matches
    url_matches = pd.merge(
        js_weekly[['ID', 'Issue #', 'Company Name', 'Clean URL', 'Clean Company', 'Industry']], 
        frontend_focus[['ID', 'Issue #', 'Company Name', 'Clean URL', 'Clean Company', 'Industry']], 
        on='Clean URL', 
        suffixes=('_JS', '_FF')
    )
    
    # Find name matches that weren't already matched by URL
    js_unmatched = js_weekly[~js_weekly['ID'].isin(url_matches['ID_JS'])]
    ff_unmatched = frontend_focus[~frontend_focus['ID'].isin(url_matches['ID_FF'])]
    
    name_matches = pd.merge(
        js_unmatched[['ID', 'Issue #', 'Company Name', 'Clean URL', 'Clean Company', 'Industry']], 
        ff_unmatched[['ID', 'Issue #', 'Company Name', 'Clean URL', 'Clean Company', 'Industry']], 
        on='Clean Company',
        suffixes=('_JS', '_FF')
    )
    
    # Filter out name matches with empty company names
    name_matches = name_matches[name_matches['Clean Company'] != ""]
    
    # Combine both match types
    all_exact_matches = pd.concat([url_matches, name_matches])
    
    return all_exact_matches

def find_fuzzy_matches(js_weekly: pd.DataFrame, frontend_focus: pd.DataFrame, 
                      exact_matches: pd.DataFrame, threshold: int = 85) -> pd.DataFrame:
    """Find fuzzy matches based on company name similarity."""
    # Get companies that haven't been exactly matched
    js_unmatched = js_weekly[
        ~js_weekly['ID'].isin(exact_matches['ID_JS']) if not exact_matches.empty else pd.Series(True, index=js_weekly.index)
    ]
    
    ff_unmatched = frontend_focus[
        ~frontend_focus['ID'].isin(exact_matches['ID_FF']) if not exact_matches.empty else pd.Series(True, index=frontend_focus.index)
    ]
    
    fuzzy_matches = []
    
    # Use tqdm for progress tracking
    total_comparisons = len(js_unmatched) * len(ff_unmatched)
    print(f"Performing {total_comparisons} fuzzy comparisons...")
    
    # Process in batches for better performance
    batch_size = 1000
    js_batches = [js_unmatched.iloc[i:i+batch_size] for i in range(0, len(js_unmatched), batch_size)]
    
    progress_bar = tqdm(total=total_comparisons)
    
    for js_batch in js_batches:
        for _, js_row in js_batch.iterrows():
            for _, ff_row in ff_unmatched.iterrows():
                name_score = fuzz.ratio(js_row['Clean Company'], ff_row['Clean Company'])
                url_score = fuzz.ratio(js_row['Clean URL'], ff_row['Clean URL']) if js_row['Clean URL'] and ff_row['Clean URL'] else 0
                
                # If either name or URL is similar enough, consider it a fuzzy match
                if name_score >= threshold or url_score >= threshold:
                    fuzzy_matches.append({
                        'ID_JS': js_row['ID'],
                        'Issue #_JS': js_row['Issue #'],
                        'Company Name_JS': js_row['Company Name'],
                        'Clean URL_JS': js_row['Clean URL'],
                        'Clean Company_JS': js_row['Clean Company'],
                        'Industry_JS': js_row['Industry'],
                        'ID_FF': ff_row['ID'],
                        'Issue #_FF': ff_row['Issue #'],
                        'Company Name_FF': ff_row['Company Name'],
                        'Clean URL_FF': ff_row['Clean URL'],
                        'Clean Company_FF': ff_row['Clean Company'],
                        'Industry_FF': ff_row['Industry'],
                        'Name Similarity': name_score,
                        'URL Similarity': url_score
                    })
                    
                progress_bar.update(1)
    
    progress_bar.close()
    
    return pd.DataFrame(fuzzy_matches) if fuzzy_matches else pd.DataFrame()

def ai_analysis_companies(model, js_unique: pd.DataFrame, ff_unique: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Use Gemini AI to analyze company relationships and find potential matches."""
    if js_unique.empty or ff_unique.empty:
        return pd.DataFrame(), {}
    
    print("Starting AI analysis of unique companies...")
    ai_matches = []
    ai_insights = {}
    
    # Select a sample of companies to avoid API rate limits
    max_companies = 50
    js_sample = js_unique.sample(min(len(js_unique), max_companies)) if len(js_unique) > max_companies else js_unique
    ff_sample = ff_unique.sample(min(len(ff_unique), max_companies)) if len(ff_unique) > max_companies else ff_unique
    
    progress_bar = tqdm(total=len(js_sample))
    
    # For each JS Weekly company, find potential matches in Frontend Focus
    for _, js_row in js_sample.iterrows():
        js_name = js_row['Company Name']
        js_url = js_row['URL']
        js_industry = js_row['Industry']
        
        # Create a list of potential matches from Frontend Focus
        ff_companies = [{"name": row['Company Name'], "url": row['URL'], "industry": row['Industry']} 
                        for _, row in ff_sample.iterrows()]
        
        # Skip if no companies to compare
        if not ff_companies:
            progress_bar.update(1)
            continue
            
        try:
            # Create prompt for Gemini
            prompt = f"""
            I need to determine if any companies in list B might be related to or the same as a company in list A.
            
            Company A:
            - Name: {js_name}
            - URL: {js_url}
            - Industry: {js_industry}
            
            List B (potential matches):
            {ff_companies[:20]}  # Limit to 20 companies to avoid token limits
            
            Analyze if any companies in list B might be:
            1. The same company as A but with different naming/branding
            2. A subsidiary or parent company of A
            3. Rebranded version of A
            4. A major partner company
            
            Return your answer as a JSON with this structure:
            {{"matches": [
                {{"ff_index": 0, "relationship": "same", "confidence": 0.9, "explanation": "explanation"}}
            ],
            "insights": "Any notable insights about company A"}}
            
            If no matches, return an empty matches array. Confidence should be between 0 and 1.
            """
            
            # Call Gemini API
            response = model.generate_content(prompt)
            result = response.text
            
            # Try to parse JSON response
            import json
            import re
            
            # Extract JSON from the response if it's wrapped in markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                result = json_match.group(1)
            
            # Clean result and parse JSON
            result = result.replace("```", "").strip()
            ai_result = json.loads(result)
            
            # Process matches
            for match in ai_result.get("matches", []):
                ff_index = match.get("ff_index")
                if 0 <= ff_index < len(ff_sample):
                    ff_row = ff_sample.iloc[ff_index]
                    ai_matches.append({
                        'ID_JS': js_row['ID'],
                        'Company Name_JS': js_name,
                        'URL_JS': js_url,
                        'Industry_JS': js_industry,
                        'ID_FF': ff_row['ID'],
                        'Company Name_FF': ff_row['Company Name'],
                        'URL_FF': ff_row['URL'],
                        'Industry_FF': ff_row['Industry'],
                        'Relationship': match.get("relationship", "unknown"),
                        'Confidence': match.get("confidence", 0),
                        'Explanation': match.get("explanation", "")
                    })
            
            # Store insights
            ai_insights[js_name] = ai_result.get("insights", "")
            
        except Exception as e:
            print(f"Error during AI analysis for {js_name}: {e}")
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    return pd.DataFrame(ai_matches) if ai_matches else pd.DataFrame(), ai_insights

def industry_categorization(model, unique_companies: pd.DataFrame) -> pd.DataFrame:
    """Use Gemini AI to categorize companies into more specific industry segments."""
    if unique_companies.empty:
        return unique_companies
    
    print("Starting AI industry categorization...")
    
    # Sample companies if the dataset is large
    max_companies = 100
    sample_companies = unique_companies.sample(min(len(unique_companies), max_companies)) if len(unique_companies) > max_companies else unique_companies
    
    results = []
    progress_bar = tqdm(total=len(sample_companies))
    
    for _, company in sample_companies.iterrows():
        try:
            # Create prompt for Gemini
            prompt = f"""
            Based on the following company information, provide a refined industry categorization:
            
            Company Name: {company['Company Name']}
            URL: {company['URL']}
            Current Industry Tag: {company['Industry']}
            
            Return your analysis as a JSON with this structure:
            {{
                "primary_industry": "Main industry category",
                "sub_category": "More specific sub-category",
                "tech_stack": ["Likely technology stack"],
                "target_audience": "Developer type (frontend, backend, fullstack, etc.)",
                "explanation": "Brief explanation of categorization"
            }}
            """
            
            # Call Gemini API
            response = model.generate_content(prompt)
            result = response.text
            
            # Parse response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                result = json_match.group(1)
            
            # Clean and parse JSON
            result = result.replace("```", "").strip()
            ai_result = json.loads(result)
            
            # Add to original data
            company_copy = company.copy()
            company_copy['Refined_Industry'] = ai_result.get('primary_industry', company['Industry'])
            company_copy['Industry_Subcategory'] = ai_result.get('sub_category', '')
            company_copy['Tech_Stack'] = ', '.join(ai_result.get('tech_stack', []))
            company_copy['Target_Audience'] = ai_result.get('target_audience', '')
            company_copy['AI_Industry_Analysis'] = ai_result.get('explanation', '')
            
            results.append(company_copy)
            
        except Exception as e:
            print(f"Error during industry categorization for {company['Company Name']}: {e}")
            results.append(company)  # Keep original data on error
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Update only the analyzed companies
    analyzed_ids = [company['ID'] for company in results]
    unanalyzed = unique_companies[~unique_companies['ID'].isin(analyzed_ids)]
    
    # Add empty columns to unanalyzed data
    for col in ['Refined_Industry', 'Industry_Subcategory', 'Tech_Stack', 'Target_Audience', 'AI_Industry_Analysis']:
        if col not in unanalyzed.columns:
            unanalyzed[col] = ''
    
    # Combine analyzed and unanalyzed data
    return pd.concat([pd.DataFrame(results), unanalyzed], ignore_index=True)

def analyze_matches(js_weekly: pd.DataFrame, frontend_focus: pd.DataFrame, 
                   exact_matches: pd.DataFrame, fuzzy_matches: pd.DataFrame,
                   ai_matches: Optional[pd.DataFrame] = None) -> Dict:
    """Generate statistics about the matches."""
    total_js = len(js_weekly)
    total_ff = len(frontend_focus)
    
    exact_match_count = len(exact_matches) if not exact_matches.empty else 0
    fuzzy_match_count = len(fuzzy_matches) if not fuzzy_matches.empty else 0
    ai_match_count = len(ai_matches) if ai_matches is not None and not ai_matches.empty else 0
    
    # Track matched company IDs
    js_matched_ids = set()
    ff_matched_ids = set()
    
    if not exact_matches.empty:
        js_matched_ids.update(exact_matches['ID_JS'])
        ff_matched_ids.update(exact_matches['ID_FF'])
    
    if not fuzzy_matches.empty:
        js_matched_ids.update(fuzzy_matches['ID_JS'])
        ff_matched_ids.update(fuzzy_matches['ID_FF'])
    
    if ai_matches is not None and not ai_matches.empty:
        js_matched_ids.update(ai_matches['ID_JS'])
        ff_matched_ids.update(ai_matches['ID_FF'])
    
    # Calculate unique companies
    js_unique_count = total_js - len(js_matched_ids)
    ff_unique_count = total_ff - len(ff_matched_ids)
    
    # Calculate match percentages
    total_matches = exact_match_count + fuzzy_match_count + ai_match_count
    js_match_percent = (len(js_matched_ids) / total_js * 100) if total_js > 0 else 0
    ff_match_percent = (len(ff_matched_ids) / total_ff * 100) if total_ff > 0 else 0
    
    # Get company distribution by industry
    industry_js = js_weekly['Industry'].value_counts().to_dict()
    industry_ff = frontend_focus['Industry'].value_counts().to_dict()
    
    # Top matched companies by industry
    matched_industries = {}
    
    def update_matched_industries(df, col_suffix):
        if df.empty:
            return
        for industry, count in df[f'Industry_{col_suffix}'].value_counts().to_dict().items():
            matched_industries[industry] = matched_industries.get(industry, 0) + count
    
    update_matched_industries(exact_matches, 'JS')
    update_matched_industries(fuzzy_matches, 'JS')
    if ai_matches is not None and not ai_matches.empty and 'Industry_JS' in ai_matches.columns:
        update_matched_industries(ai_matches, 'JS')
    
    return {
        'total_js': total_js,
        'total_ff': total_ff,
        'exact_matches': exact_match_count,
        'fuzzy_matches': fuzzy_match_count,
        'ai_matches': ai_match_count,
        'total_matches': total_matches,
        'js_unique': js_unique_count,
        'ff_unique': ff_unique_count,
        'js_match_percent': js_match_percent,
        'ff_match_percent': ff_match_percent,
        'industry_js': industry_js,
        'industry_ff': industry_ff,
        'matched_industries': matched_industries,
        'js_matched_ids': js_matched_ids,
        'ff_matched_ids': ff_matched_ids
    }

def generate_advanced_visualizations(stats: Dict, js_weekly: pd.DataFrame, 
                                   frontend_focus: pd.DataFrame, output_dir: str = './'):
    """Generate enhanced visualizations of the analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Enhanced match distribution pie chart
    labels = ['Exact Matches', 'Fuzzy Matches', 'AI-detected Matches', 'JS Weekly Only', 'Frontend Focus Only']
    sizes = [
        stats['exact_matches'], 
        stats['fuzzy_matches'], 
        stats.get('ai_matches', 0),
        stats['js_unique'], 
        stats['ff_unique']
    ]
    
    # Check if we have any non-zero values
    if sum(sizes) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=None,  # We'll add a legend instead
            autopct='%1.1f%%', 
            startangle=90,
            shadow=True,
            explode=(0.05, 0.05, 0.05, 0, 0)  # Emphasize the matches
        )
        
        # Customize appearance
        plt.setp(autotexts, size=10, weight="bold")
        ax.axis('equal')
        ax.legend(wedges, labels, title="Match Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.title('Distribution of Sponsor Matches Between Newsletters', fontsize=14, fontweight='bold')
        plt.savefig(output_path / 'match_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Enhanced industry comparison bar chart
    js_industries = stats['industry_js']
    ff_industries = stats['industry_ff']
    
    # Get all unique industries
    all_industries = set(js_industries.keys()).union(set(ff_industries.keys()))
    
    # Prepare data for plotting
    industry_data = {
        'Industry': list(all_industries),
        'JS Weekly': [js_industries.get(ind, 0) for ind in all_industries],
        'Frontend Focus': [ff_industries.get(ind, 0) for ind in all_industries],
        'Total': [js_industries.get(ind, 0) + ff_industries.get(ind, 0) for ind in all_industries]
    }
    
    industry_df = pd.DataFrame(industry_data)
    industry_df = industry_df.sort_values('Total', ascending=False).head(10)  # Top 10 industries
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ind = np.arange(len(industry_df))
    width = 0.35
    
    # Create bars with custom styling
    js_bars = ax.bar(ind - width/2, industry_df['JS Weekly'], width, label='JS Weekly', 
                   color='#3498db', edgecolor='black', linewidth=1.2, alpha=0.8)
    ff_bars = ax.bar(ind + width/2, industry_df['Frontend Focus'], width, label='Frontend Focus', 
                   color='#e74c3c', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Add data labels on top of bars
    for bar in js_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
                
    for bar in ff_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Customize appearance
    ax.set_xticks(ind)
    ax.set_xticklabels(industry_df['Industry'], rotation=45, ha='right', fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylabel('Number of Companies', fontweight='bold', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.title('Top 10 Industries by Newsletter', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'industry_comparison.png', dpi=300)
    plt.close()
    
    # 3. Match percentage by industry chart
    if stats['matched_industries']:
        # Calculate match percentages by industry
        match_data = []
        for industry, matched_count in stats['matched_industries'].items():
            js_count = js_industries.get(industry, 0)
            ff_count = ff_industries.get(industry, 0)
            if js_count > 0 and ff_count > 0:
                match_percent_js = (matched_count / js_count) * 100
                match_percent_ff = (matched_count / ff_count) * 100
                match_data.append({
                    'Industry': industry,
                    'Match % in JS Weekly': match_percent_js,
                    'Match % in Frontend Focus': match_percent_ff,
                    'Total Companies': js_count + ff_count
                })
        
        if match_data:
            match_df = pd.DataFrame(match_data)
            match_df = match_df.sort_values('Total Companies', ascending=False).head(8)  # Top 8 industries
            
            fig, ax = plt.subplots(figsize=(14, 10))
            ind = np.arange(len(match_df))
            width = 0.35
            
            ax.bar(ind - width/2, match_df['Match % in JS Weekly'], width, label='JS Weekly', 
                  color='#3498db', edgecolor='black', alpha=0.8)
            ax.bar(ind + width/2, match_df['Match % in Frontend Focus'], width, label='Frontend Focus', 
                  color='#e74c3c', edgecolor='black', alpha=0.8)
            
            ax.set_xticks(ind)
            ax.set_xticklabels(match_df['Industry'], rotation=45, ha='right', fontweight='bold')
            ax.set_ylabel('Match Percentage (%)', fontweight='bold')
            ax.set_title('Match Percentage by Industry', fontsize=16, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(output_path / 'match_percentage_by_industry.png', dpi=300)
            plt.close()
    
    # 4. Company overlap visualization (Venn-like)
    try:
        from matplotlib_venn import venn2
        
        # Create Venn diagram
        fig, ax = plt.subplots(figsize=(10, 10))
        v = venn2(subsets=(stats['js_unique'], stats['ff_unique'], stats['total_matches']), 
                set_labels=('JS Weekly', 'Frontend Focus'))
        
        # Customize appearance
        v.get_patch_by_id('10').set_color('#3498db')
        v.get_patch_by_id('01').set_color('#e74c3c')
        v.get_patch_by_id('11').set_color('#2ecc71')
        
        plt.title('Company Overlap Between Newsletters', fontsize=16, fontweight='bold')
        plt.savefig(output_path / 'company_overlap_venn.png', dpi=300, bbox_inches='tight')
        plt.close()
    except ImportError:
        print("matplotlib_venn not installed. Skipping Venn diagram.")

def generate_enhanced_reports(js_weekly: pd.DataFrame, frontend_focus: pd.DataFrame, 
                            exact_matches: pd.DataFrame, fuzzy_matches: pd.DataFrame,
                            ai_matches: pd.DataFrame, ai_insights: Dict,
                            stats: Dict, output_dir: str = './'):
    """Generate enhanced CSV and HTML reports of the analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract IDs of matched companies
    js_matched_ids = stats['js_matched_ids']
    ff_matched_ids = stats['ff_matched_ids']
    
    # 1. Save match files
    if not exact_matches.empty:
        exact_matches.to_csv(output_path / 'exact_matches.csv', index=False)
    
    if not fuzzy_matches.empty:
        fuzzy_matches.to_csv(output_path / 'fuzzy_matches.csv', index=False)
    
    if not ai_matches.empty:
        ai_matches.to_csv(output_path / 'ai_matches.csv', index=False)
    
    # 2. Get companies unique to each newsletter
    js_unique = js_weekly[~js_weekly['ID'].isin(js_matched_ids)]
    ff_unique = frontend_focus[~frontend_focus['ID'].isin(ff_matched_ids)]
    
    # 3. Save unique companies
    js_unique.to_csv(output_path / 'js_weekly_unique.csv', index=False)
    ff_unique.to_csv(output_path / 'frontend_focus_unique.csv', index=False)
    
    # 4. Save AI insights
    if ai_insights:
        with open(output_path / 'ai_company_insights.txt', 'w') as f:
            f.write("AI Insights for Companies\n")
            f.write("=======================\n\n")
            for company, insight in ai_insights.items():
                f.write(f"Company: {company}\n")
                f.write(f"Insight: {insight}\n")
                f.write("-" * 50 + "\n\n")
    
    # 5. Enhanced summary statistics
    with open(output_path / 'enhanced_analytics_summary.txt', 'w') as f:
        f.write(f"Advanced Newsletter Sponsor Analysis Summary\n")
        f.write(f"==========================================\n\n")
        f.write(f"Total JavaScript Weekly sponsors: {stats['total_js']}\n")
        f.write(f"Total Frontend Focus sponsors: {stats['total_ff']}\n\n")
        
        f.write(f"Match Analysis:\n")
        f.write(f"  - Exact matches: {stats['exact_matches']} ({stats['exact_matches']/max(1,stats['total_matches'])*100:.1f}% of all matches)\n")
        f.write(f"  - Fuzzy matches: {stats['fuzzy_matches']} ({stats['fuzzy_matches']/max(1,stats['total_matches'])*100:.1f}% of all matches)\n")
        f.write(f"  - AI-detected matches: {stats['ai_matches']} ({stats['ai_matches']/max(1,stats['total_matches'])*100:.1f}% of all matches)\n")
        f.write(f"  - Total matches: {stats['total_matches']}\n\n")
        
        f.write(f"Match Percentages:\n")
        f.write(f"  - JS Weekly companies in Frontend Focus: {stats['js_match_percent']:.1f}%\n")
        f.write(f"  - Frontend Focus companies in JS Weekly: {stats['ff_match_percent']:.1f}%\n\n")
        
        f.write(f"Unique Companies:\n")
        f.write(f"  - Companies unique to JS Weekly: {stats['js_unique']} ({stats['js_unique']/stats['total_js']*100:.1f}% of JS Weekly)\n")
        f.write(f"  - Companies unique to Frontend Focus: {stats['ff_unique']} ({stats['ff_unique']/stats['total_ff']*100:.1f}% of Frontend Focus)\n\n")
        
        f.write(f"Top Industries in JS Weekly:\n")
        for ind, count in sorted(stats['industry_js'].items(), key=lambda x: x[1], reverse=True)[:5]:
            f.write(f"  - {ind}: {count} ({count/stats['total_js']*100:.1f}%)\n")
        
        f.write(f"\nTop Industries in Frontend Focus:\n")
        for ind, count in sorted(stats['industry_ff'].items(), key=lambda x: x[1], reverse=True)[:5]:
            f.write(f"  - {ind}: {count} ({count/stats['total_ff']*100:.1f}%)\n")
        
        f.write(f"\nTop Matched Industries:\n")
        for ind, count in sorted(stats['matched_industries'].items(), key=lambda x: x[1], reverse=True)[:5]:
            js_count = stats['industry_js'].get(ind, 0)
            ff_count = stats['industry_ff'].get(ind, 0)
            f.write(f"  - {ind}: {count} matches ({count/max(1, js_count)*100:.1f}% of JS Weekly {ind}, {count/max(1, ff_count)*100:.1f}% of FF {ind})\n")
    
    # 6. Create HTML summary report
    create_html_summary_report(
        stats, 
        js_weekly, 
        frontend_focus, 
        exact_matches, 
        fuzzy_matches, 
        ai_matches,
        output_path / 'newsletter_analysis_report.html'
    )

def create_html_summary_report(stats: Dict, js_weekly: pd.DataFrame, frontend_focus: pd.DataFrame,
                             exact_matches: pd.DataFrame, fuzzy_matches: pd.DataFrame,
                             ai_matches: pd.DataFrame, output_path: Path):
    """Create an HTML report summarizing the analysis."""
    
    # Function to create HTML table from DataFrame
    def df_to_html_table(df, max_rows=10):
        if df.empty:
            return "<p>No data available</p>"
        
        # Limit rows for display
        display_df = df.head(max_rows)
        
        # Convert to HTML table with Bootstrap styling
        html = display_df.to_html(classes="table table-striped table-hover table-sm", index=False)
        
        if len(df) > max_rows:
            html += f"<p><em>Showing {max_rows} of {len(df)} rows</em></p>"
            
        return html
    
    # HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Newsletter Sponsor Analysis Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            .stats-card {{ margin-bottom: 20px; }}
            .chart-container {{ margin: 30px 0; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .highlight {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            .badge-primary {{ background-color: #3498db; }}
            .badge-secondary {{ background-color: #e74c3c; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4 mb-4">Newsletter Sponsor Analysis Report</h1>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card stats-card">
                        <div class="card-header bg-primary text-white">
                            <h3 class="card-title mb-0">JS Weekly</h3>
                        </div>
                        <div class="card-body">
                            <p><strong>Total Companies:</strong> {stats['total_js']}</p>
                            <p><strong>Matched Companies:</strong> {len(stats['js_matched_ids'])} ({stats['js_match_percent']:.1f}%)</p>
                            <p><strong>Unique Companies:</strong> {stats['js_unique']} ({stats['js_unique']/stats['total_js']*100:.1f}%)</p>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card stats-card">
                        <div class="card-header bg-danger text-white">
                            <h3 class="card-title mb-0">Frontend Focus</h3>
                        </div>
                        <div class="card-body">
                            <p><strong>Total Companies:</strong> {stats['total_ff']}</p>
                            <p><strong>Matched Companies:</strong> {len(stats['ff_matched_ids'])} ({stats['ff_match_percent']:.1f}%)</p>
                            <p><strong>Unique Companies:</strong> {stats['ff_unique']} ({stats['ff_unique']/stats['total_ff']*100:.1f}%)</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h3 class="card-title mb-0">Match Analysis</h3>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-4">
                                    <div class="highlight text-center">
                                        <h4>Exact Matches</h4>
                                        <h2>{stats['exact_matches']}</h2>
                                        <p>({stats['exact_matches']/max(1,stats['total_matches'])*100:.1f}% of matches)</p>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="highlight text-center">
                                        <h4>Fuzzy Matches</h4>
                                        <h2>{stats['fuzzy_matches']}</h2>
                                        <p>({stats['fuzzy_matches']/max(1,stats['total_matches'])*100:.1f}% of matches)</p>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="highlight text-center">
                                        <h4>AI-detected Matches</h4>
                                        <h2>{stats['ai_matches']}</h2>
                                        <p>({stats['ai_matches']/max(1,stats['total_matches'])*100:.1f}% of matches)</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Visualization References</h3>
                <p>Charts have been saved to the output directory:</p>
                <ul>
                    <li>Company Distribution: <code>match_distribution.png</code></li>
                    <li>Industry Comparison: <code>industry_comparison.png</code></li>
                    <li>Match Percentage by Industry: <code>match_percentage_by_industry.png</code></li>
                    <li>Company Overlap: <code>company_overlap_venn.png</code></li>
                </ul>
            </div>
            
            <h2 class="mt-5">Sample Data</h2>
            
            <div class="mt-4">
                <h3>Exact Matches <span class="badge bg-primary">{stats['exact_matches']}</span></h3>
                {df_to_html_table(exact_matches)}
            </div>
            
            <div class="mt-5">
                <h3>Fuzzy Matches <span class="badge bg-primary">{stats['fuzzy_matches']}</span></h3>
                {df_to_html_table(fuzzy_matches)}
            </div>
            
            <div class="mt-5">
                <h3>AI-detected Matches <span class="badge bg-primary">{stats['ai_matches']}</span></h3>
                {df_to_html_table(ai_matches)}
            </div>
            
            <div class="mt-5">
                <h3>Top JS Weekly Unique Companies</h3>
                {df_to_html_table(js_weekly[~js_weekly['ID'].isin(stats['js_matched_ids'])])}
            </div>
            
            <div class="mt-5">
                <h3>Top Frontend Focus Unique Companies</h3>
                {df_to_html_table(frontend_focus[~frontend_focus['ID'].isin(stats['ff_matched_ids'])])}
            </div>
            
            <div class="mt-5 mb-5">
                <h3>Industry Distribution Comparison</h3>
                <table class="table table-bordered table-hover">
                    <thead>
                        <tr>
                            <th>Industry</th>
                            <th>JS Weekly</th>
                            <th>Frontend Focus</th>
                            <th>Match Count</th>
                            <th>Match Rate</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add industry data
    all_industries = set(stats['industry_js'].keys()).union(set(stats['industry_ff'].keys()))
    for industry in sorted(all_industries, key=lambda x: stats['industry_js'].get(x, 0) + stats['industry_ff'].get(x, 0), reverse=True)[:15]:
        js_count = stats['industry_js'].get(industry, 0)
        ff_count = stats['industry_ff'].get(industry, 0)
        match_count = stats['matched_industries'].get(industry, 0)
        match_rate = match_count / max(1, min(js_count, ff_count)) * 100
        
        html_content += f"""
                        <tr>
                            <td>{industry}</td>
                            <td>{js_count}</td>
                            <td>{ff_count}</td>
                            <td>{match_count}</td>
                            <td>{match_rate:.1f}%</td>
                        </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <footer class="bg-light text-center text-muted py-4">
            <p>Generated with Enhanced Sponsor Analysis Tool with Gemini AI</p>
        </footer>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Enhanced sponsor data analysis between JS Weekly and Frontend Focus')
    parser.add_argument('--js', required=True, help='Path to JavaScript Weekly Excel file')
    parser.add_argument('--ff', required=True, help='Path to Frontend Focus Excel file')
    parser.add_argument('--output', default='./results', help='Output directory for results')
    parser.add_argument('--threshold', type=int, default=85, help='Similarity threshold for fuzzy matching (0-100)')
    parser.add_argument('--gemini-key', help='Google Gemini API key for AI analysis')
    parser.add_argument('--skip-ai', action='store_true', help='Skip AI-powered analysis')
    parser.add_argument('--advanced-categorization', action='store_true', help='Use AI for advanced industry categorization')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Loading data...")
    # Load data
    js_weekly, frontend_focus = load_data(args.js, args.ff)
    print(f"Loaded {len(js_weekly)} JS Weekly entries and {len(frontend_focus)} Frontend Focus entries")
    
    # Find matches
    print("Finding exact matches...")
    exact_matches = find_exact_matches(js_weekly, frontend_focus)
    print(f"Found {len(exact_matches)} exact matches")
    
    print("Finding fuzzy matches...")
    fuzzy_matches = find_fuzzy_matches(js_weekly, frontend_focus, exact_matches, args.threshold)
    print(f"Found {len(fuzzy_matches)} fuzzy matches")
    
    # Initialize variables for AI analysis
    ai_matches = pd.DataFrame()
    ai_insights = {}
    gemini_model = None
    
    # Configure Gemini AI if API key is provided and AI analysis is not skipped
    if args.gemini_key and not args.skip_ai:
        try:
            print("Configuring Gemini AI...")
            gemini_model = configure_gemini(args.gemini_key)
            
            # Get companies that haven't been matched yet
            js_matched_ids = set()
            ff_matched_ids = set()
            
            if not exact_matches.empty:
                js_matched_ids.update(exact_matches['ID_JS'])
                ff_matched_ids.update(exact_matches['ID_FF'])
            
            if not fuzzy_matches.empty:
                js_matched_ids.update(fuzzy_matches['ID_JS'])
                ff_matched_ids.update(fuzzy_matches['ID_FF'])
            
            js_unique = js_weekly[~js_weekly['ID'].isin(js_matched_ids)]
            ff_unique = frontend_focus[~frontend_focus['ID'].isin(ff_matched_ids)]
            
            # Perform AI analysis
            print("Performing AI analysis to find additional matches...")
            ai_matches, ai_insights = ai_analysis_companies(gemini_model, js_unique, ff_unique)
            print(f"AI analysis found {len(ai_matches)} potential additional matches")
            
            # Perform advanced industry categorization if requested
            if args.advanced_categorization:
                print("Performing advanced industry categorization...")
                js_unique = industry_categorization(gemini_model, js_unique)
                js_unique.to_csv(output_dir / 'js_weekly_categorized.csv', index=False)
                print(f"Saved categorized JS Weekly data to {output_dir / 'js_weekly_categorized.csv'}")
        
        except Exception as e:
            print(f"Error during AI analysis: {e}")
            print("Continuing with standard analysis...")
    
    # Analyze matches including AI matches if available
    print("Analyzing matches...")
    stats = analyze_matches(js_weekly, frontend_focus, exact_matches, fuzzy_matches, ai_matches)
    
    # Generate enhanced reports and visualizations
    print("Generating reports and visualizations...")
    generate_enhanced_reports(
        js_weekly, frontend_focus, exact_matches, fuzzy_matches, 
        ai_matches, ai_insights, stats, args.output
    )
    generate_advanced_visualizations(stats, js_weekly, frontend_focus, args.output)
    
    print(f"Analysis complete! Enhanced results saved to {args.output}")
    print(f"Found {stats['exact_matches']} exact matches, {stats['fuzzy_matches']} fuzzy matches, and {stats.get('ai_matches', 0)} AI-detected matches")
    print(f"Overall match percentage: JS Weekly {stats['js_match_percent']:.1f}%, Frontend Focus {stats['ff_match_percent']:.1f}%")
    print(f"Check the HTML report at {output_dir / 'newsletter_analysis_report.html'} for a comprehensive overview")

if __name__ == "__main__":
    main()

# python newsletter_comparison.py --js your/file/path/to/js.xlsx --ff your/file/path/to/ff.xlsx --output ./results --gemini-key YOUR_API_KEY
