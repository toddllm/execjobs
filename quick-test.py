import json
import pickle
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List
import re
import requests
from bs4 import BeautifulSoup
import time

def get_filing_text(filing: Dict) -> str:
    """Get the text content of a filing"""
    try:
        if not all(k in filing for k in ['cik', 'accession']):
            return ""
            
        accession_formatted = filing['accession'].replace('-', '')
        url = f"https://www.sec.gov/Archives/edgar/data/{filing['cik']}/{accession_formatted}/{filing['accession']}.txt"
        
        headers = {
            'User-Agent': 'Company Research todd.deshane@gmail.com',
            'Accept': 'application/json'
        }
        
        # Add delay to respect SEC rate limits
        time.sleep(0.1)
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            # Parse the filing text
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            # Clean up text
            text = ' '.join(text.split())
            return text
        else:
            print(f"Failed to fetch filing text. Status code: {response.status_code}")
            return ""
            
    except Exception as e:
        print(f"Error getting filing text: {str(e)}")
        return ""

def analyze_sample_filings():
    """Analyze a sample of cached filings to check AI-related content"""
    cache_dir = Path('sec_cache')
    filings_dir = cache_dir / 'filings'
    financials_dir = cache_dir / 'financials'
    
    # Sample 3 companies that have both filings and financials
    sample_companies = []
    for filing_path in list(filings_dir.glob('*.pickle'))[:3]:
        cik = filing_path.stem
        if (financials_dir / f"{cik}.json").exists():
            sample_companies.append(cik)
    
    results = []
    
    # AI-related keywords to look for (expanded list)
    ai_keywords = [
        'artificial intelligence', 'machine learning', 'AI initiatives',
        'AI adoption', 'large language model', 'generative AI',
        'neural network', 'deep learning', 'AI technology',
        'AI capabilities', 'AI solutions', 'AI platform',
        'AI-powered', 'AI applications', 'AI tools'
    ]
    
    print("\nAnalyzing sample of current data collection...")
    
    for cik in sample_companies:
        # Load financial data
        with open(financials_dir / f"{cik}.json", 'r') as f:
            financial_data = json.load(f)
            
        # Load filings
        with open(filings_dir / f"{cik}.pickle", 'rb') as f:
            filings = pickle.load(f)
        
        # Analyze one recent 10-K filing for this company
        recent_10k = None
        for filing in sorted(filings, key=lambda x: x.get('filing_date', ''), reverse=True):
            if filing.get('form') == '10-K':
                recent_10k = filing
                break
        
        if recent_10k:
            company_result = {
                'cik': cik,
                'filing_date': recent_10k.get('filing_date'),
                'revenue_data_points': len(financial_data.get('revenue', [])),
                'has_rd_data': bool(financial_data.get('rd_expenses')),
                'ai_mentions': {},
                'recent_revenue': None,
                'filing_accession': recent_10k.get('accession')
            }
            
            # Get most recent revenue
            if financial_data.get('revenue'):
                recent_revenue = sorted(
                    financial_data['revenue'], 
                    key=lambda x: x.get('end', ''), 
                    reverse=True
                )
                if recent_revenue:
                    company_result['recent_revenue'] = recent_revenue[0].get('val')
            
            # Get and check filing text for AI mentions
            filing_text = get_filing_text(recent_10k).lower()
            
            if filing_text:
                print(f"\nAnalyzing filing text for CIK {cik} ({len(filing_text)} characters)")
                for keyword in ai_keywords:
                    matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', filing_text))
                    if matches > 0:
                        company_result['ai_mentions'][keyword] = matches
                
                # Save a small sample of text around first AI mention (if any)
                for keyword, count in company_result['ai_mentions'].items():
                    match = re.search(r'.{100}' + re.escape(keyword) + r'.{100}', filing_text)
                    if match:
                        company_result['sample_context'] = '...' + match.group(0) + '...'
                        break
            
            results.append(company_result)
    
    print("\nSample Analysis Results:")
    print("------------------------")
    
    for result in results:
        print(f"\nCompany CIK: {result['cik']}")
        print(f"Latest 10-K Date: {result['filing_date']}")
        print(f"Filing Accession: {result['filing_accession']}")
        print(f"Revenue Data Points: {result['revenue_data_points']}")
        print(f"Has R&D Data: {result['has_rd_data']}")
        print(f"Recent Revenue: ${result['recent_revenue']:,.0f}" if result['recent_revenue'] else "No recent revenue")
        
        if result['ai_mentions']:
            print("AI-Related Mentions:")
            for keyword, count in result['ai_mentions'].items():
                print(f"- '{keyword}': {count} mentions")
            if result.get('sample_context'):
                print("\nSample context:")
                print(result['sample_context'])
        else:
            print("No AI-related mentions found in recent 10-K")
    
    # Assessment
    print("\nAssessment:")
    print("-----------")
    
    # Check if we have enough data for AI adoption analysis
    has_revenue = all(r['recent_revenue'] is not None for r in results)
    has_ai_mentions = any(r['ai_mentions'] for r in results)
    
    if has_revenue and has_ai_mentions:
        print("✅ Current data collection appears sufficient for AI adoption analysis:")
        print("   - Financial data is being captured")
        print("   - AI mentions are detectable in filings")
        print("   - Can proceed with current collection and do post-processing")
    else:
        print("⚠️ Data collection assessment:")
        if not has_revenue:
            print("   - Missing revenue data for some companies")
        else:
            print("   + Revenue data looks good")
            
        if not has_ai_mentions:
            print("   - No AI mentions detected (checking actual filing text now)")
        else:
            print("   + Successfully detecting AI mentions")
    
    # R&D data assessment
    if not any(r['has_rd_data'] for r in results):
        print("\nNote: R&D data is missing but not critical if we can detect AI initiatives through filings")

if __name__ == "__main__":
    analyze_sample_filings()