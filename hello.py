# Import all required libraries at the top level
import pandas as pd
import numpy as np
import requests
import json
import re
from datetime import datetime
from rich.console import Console
from rich.progress import track
import time
import random
from pathlib import Path
import os
from typing import Dict, List, Tuple
import logging
import argparse
import pickle
import yfinance as yf
from bs4 import BeautifulSoup

def setup_logger(verbose: bool = False) -> logging.Logger:
    """Set up and configure logger"""
    logger = logging.getLogger('SECAnalyzer')
    
    # Set logging level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger if it doesn't already have one
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger

class DataCache:
    def __init__(self, cache_dir: str = 'sec_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / 'financials').mkdir(exist_ok=True)
        (self.cache_dir / 'filings').mkdir(exist_ok=True)
        (self.cache_dir / 'metadata').mkdir(exist_ok=True)
    
    def get_financial_data(self, cik: str) -> Dict:
        """Get financial data from cache if it exists"""
        cache_file = self.cache_dir / 'financials' / f'{cik}.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_financial_data(self, cik: str, data: Dict):
        """Save financial data to cache"""
        cache_file = self.cache_dir / 'financials' / f'{cik}.json'
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    def get_filings(self, cik: str) -> Dict:
        """Get filings from cache if they exist"""
        cache_file = self.cache_dir / 'filings' / f'{cik}.pickle'
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_filings(self, cik: str, filings: List[Dict]):
        """Save filings to cache"""
        cache_file = self.cache_dir / 'filings' / f'{cik}.pickle'
        with open(cache_file, 'wb') as f:
            pickle.dump(filings, f)

class RateLimiter:
    def __init__(self, min_delay: float = 1.0, max_delay: float = 3.0, backoff_factor: float = 2):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.last_request_time = 0
        self.current_delay = min_delay
        self.consecutive_failures = 0
        self.requests_count = 0
        self.start_time = time.time()
    
    def wait(self):
        """Wait a random amount of time between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Calculate requests per minute
        elapsed_minutes = (current_time - self.start_time) / 60
        if elapsed_minutes > 0:
            rpm = self.requests_count / elapsed_minutes
            if rpm > 8:  # If more than 8 requests per minute
                self.current_delay = min(self.max_delay, self.current_delay * 1.5)
        
        # Always wait at least min_delay between requests
        delay = max(0, self.current_delay - time_since_last)
        if delay > 0:
            actual_delay = random.uniform(delay, delay * 1.5)
            time.sleep(actual_delay)
        
        self.last_request_time = time.time()
        self.requests_count += 1
    
    def success(self):
        """Call this when a request succeeds to gradually reduce delay"""
        self.consecutive_failures = 0
        if self.current_delay > self.min_delay:
            self.current_delay = max(
                self.min_delay,
                self.current_delay / self.backoff_factor
            )
    
    def failure(self):
        """Call this when a request fails to increase delay"""
        self.consecutive_failures += 1
        self.current_delay = min(
            self.max_delay * (self.backoff_factor ** self.consecutive_failures),
            10.0  # Hard cap at 10 seconds
        )

class SECDataFetcher:
    def __init__(self, logger: logging.Logger, cache: DataCache, force_download: bool = False):
        self.base_url = "https://data.sec.gov"
        self.company_facts_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"
        self.submissions_url = "https://data.sec.gov/submissions/CIK{}.json"
        self.headers = {
            'User-Agent': 'Company Research todd.deshane@gmail.com',
            'Accept': 'application/json'
        }
        self.logger = logger
        self.cache = cache
        self.force_download = force_download
        # More conservative rate limiting
        self.rate_limiter = RateLimiter(
            min_delay=2.0,     # Minimum 2 seconds between requests
            max_delay=5.0,     # Maximum 5 seconds between requests
            backoff_factor=1.5  # Gentle backoff
        )

    def get_financial_data(self, cik: str) -> Dict:
        """Fetch financial data with caching"""
        if not self.force_download:
            cached_data = self.cache.get_financial_data(cik)
            if cached_data is not None:
                self.logger.debug(f"Using cached financial data for CIK {cik}")
                return cached_data
        
        try:
            self.rate_limiter.wait()
            padded_cik = cik.zfill(10)
            url = self.company_facts_url.format(padded_cik)
            
            self.logger.debug(f"Fetching financial data from: {url}")
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                self.rate_limiter.success()
                data = response.json()
                metrics = self._process_financial_data(data)
                
                # Cache the results
                self.cache.save_financial_data(cik, metrics)
                return metrics
            else:
                self.rate_limiter.failure()
                self.logger.warning(f"Failed to fetch data for CIK {cik}. Status code: {response.status_code}")
                return self._get_empty_metrics()
            
        except Exception as e:
            self.logger.error(f"Error fetching financial data: {str(e)}")
            return self._get_empty_metrics()

    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics dictionary"""
        return {
            'revenue': [],
            'rd_expenses': [],
            'operating_expenses': [],
            'net_income': []
        }

    def _process_financial_data(self, data: Dict) -> Dict:
        """Process raw financial data"""
        metrics = self._get_empty_metrics()
        
        if 'facts' in data and 'us-gaap' in data['facts']:
            us_gaap = data['facts']['us-gaap']
            metrics = {
                'revenue': self._extract_metric(us_gaap, ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax']),
                'rd_expenses': self._extract_metric(us_gaap, ['ResearchAndDevelopmentExpense']),
                'operating_expenses': self._extract_metric(us_gaap, ['OperatingExpenses', 'CostsAndExpenses']),
                'net_income': self._extract_metric(us_gaap, ['NetIncomeLoss'])
            }
        
        return metrics

    def _extract_metric(self, us_gaap: Dict, possible_keys: List[str]) -> List[Dict]:
        """Extract metric from us-gaap data using possible keys"""
        for key in possible_keys:
            if key in us_gaap:
                return self._process_units(us_gaap[key])
        return []
    
    def _process_units(self, metric_data: Dict) -> List[Dict]:
        """Process units from metric data"""
        if 'units' not in metric_data:
            return []
            
        units = metric_data['units']
        if 'USD' in units:
            return units['USD']
        return []

    def get_filings_text(self, cik: str) -> List[Dict]:
        """Fetch filings with caching"""
        if not self.force_download:
            cached_filings = self.cache.get_filings(cik)
            if cached_filings is not None:
                self.logger.debug(f"Using cached filings for CIK {cik}")
                return cached_filings
        
        try:
            self.rate_limiter.wait()
            padded_cik = cik.zfill(10)
            url = self.submissions_url.format(padded_cik)
            
            self.logger.debug(f"Fetching filings from: {url}")
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                self.rate_limiter.success()
                filings = self._process_filings(response.json(), cik)  # Pass CIK to process_filings
                self.cache.save_filings(cik, filings)
                return filings
            else:
                self.rate_limiter.failure()
                self.logger.warning(f"Failed to fetch filings for CIK {cik}. Status code: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching filings for CIK {cik}: {str(e)}")
            return []

    def _process_filings(self, data: Dict, cik: str) -> List[Dict]:
        """Process raw filings data"""
        if 'filings' not in data or 'recent' not in data['filings']:
            return []
            
        filings = []
        recent = data['filings']['recent']
        
        # Get the fields we need
        accessions = recent.get('accessionNumber', [])
        forms = recent.get('form', [])
        filing_dates = recent.get('filingDate', [])
        report_dates = recent.get('reportDate', [])
        
        # Zip them together
        for acc, form, f_date, r_date in zip(accessions, forms, filing_dates, report_dates):
            if form in ['10-K', '10-Q']:  # Only include 10-K and 10-Q filings
                filings.append({
                    'accession': acc,
                    'form': form,
                    'filing_date': f_date,
                    'report_date': r_date,
                    'cik': cik  # Add CIK to each filing
                })
                self.logger.debug(f"Added {form} filing from {f_date} for CIK {cik}")
        
        return filings

class AIAdoptionAnalyzer:
    def __init__(self, fetcher: SECDataFetcher, logger: logging.Logger):
        self.fetcher = fetcher
        self.logger = logger
        self.ai_keywords = {
            'strategic_initiatives': [
                'digital transformation', 'AI strategy', 'AI initiatives',
                'artificial intelligence strategy', 'machine learning strategy',
                'AI investment', 'AI adoption', 'AI implementation'
            ],
            'early_stage': [
                'exploring AI', 'investigating AI', 'evaluating AI',
                'AI pilot', 'AI prototype', 'AI experiment',
                'planning to implement', 'beginning to implement'
            ],
            'infrastructure': [
                'data infrastructure', 'cloud migration', 'AI readiness',
                'machine learning infrastructure', 'data strategy',
                'modernizing systems', 'digital infrastructure'
            ],
            'talent': [
                'AI talent', 'machine learning engineer', 'data scientist',
                'AI capabilities', 'AI expertise', 'technical talent',
                'hiring in AI', 'AI skills'
            ],
            'specific_tech': [
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural network', 'AI', 'ML', 'computer vision',
                'natural language processing', 'NLP', 'large language model',
                'generative AI', 'predictive analytics'
            ]
        }
    
    def _load_company_list(self) -> List[Dict]:
        """Load S&P 500 companies and their CIK numbers"""
        try:
            # Get list of current S&P 500 constituents
            sp500_components = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            
            # Get CIK mapping from SEC
            response = requests.get(
                'https://www.sec.gov/files/company_tickers.json',
                headers=self.fetcher.headers
            )
            
            if response.status_code != 200:
                raise Exception("Failed to fetch SEC CIK data")
                
            cik_data = response.json()
            
            # Create CIK lookup dictionary
            cik_lookup = {
                company['ticker']: str(company['cik_str'])
                for company in cik_data.values()
            }
            
            companies = []
            for _, row in sp500_components.iterrows():
                symbol = row['Symbol']
                if symbol in cik_lookup:
                    companies.append({
                        'Symbol': symbol,
                        'Name': row['Security'],
                        'CIK': cik_lookup[symbol],
                        'Industry': row['GICS Sector']
                    })
                    self.logger.debug(f"Added {symbol} (CIK: {cik_lookup[symbol]}) to company list")
                else:
                    self.logger.warning(f"No CIK found for {symbol} in SEC database")
            
            self.logger.info(f"Loaded {len(companies)} companies from S&P 500")
            return companies
            
        except Exception as e:
            self.logger.error(f"Error loading company list: {str(e)}")
            # Return sample companies as fallback
            return [
                {'Symbol': 'AAPL', 'Name': 'Apple Inc.', 'CIK': '320193', 'Industry': 'Technology'},
                {'Symbol': 'MSFT', 'Name': 'Microsoft Corporation', 'CIK': '789019', 'Industry': 'Technology'},
                {'Symbol': 'GOOGL', 'Name': 'Alphabet Inc.', 'CIK': '1652044', 'Industry': 'Technology'}
            ]
    
    def run_analysis(self) -> pd.DataFrame:
        """Run the main analysis pipeline"""
        # Load company list
        companies = self._load_company_list()
        
        results = []
        for company in track(companies, description="Analyzing companies..."):
            try:
                company_data = self._analyze_company(company)
                if company_data:
                    results.append(company_data)
            except Exception as e:
                self.logger.error(f"Error analyzing {company['Symbol']}: {str(e)}")
        
        return pd.DataFrame(results)

    def _analyze_company(self, company: Dict) -> Dict:
        """Analyze a single company"""
        cik = company['CIK']
        
        # Get financial data and filings
        financials = self.fetcher.get_financial_data(cik)
        filings = self.fetcher.get_filings_text(cik)
        
        if not financials or not filings:
            return None
        
        # Calculate metrics
        latest_revenue = self._get_latest_metric(financials['revenue'])
        latest_rd = self._get_latest_metric(financials['rd_expenses'])
        ai_analysis = self._analyze_ai_mentions(filings)
        
        # Calculate AI readiness score
        ai_readiness = self._calculate_ai_readiness(
            ai_analysis['adoption_indicators'],
            latest_rd / latest_revenue if latest_revenue else 0
        )
        
        # Analyze quarterly trends
        quarters = sorted(ai_analysis['quarterly_trends'].keys())
        recent_trend = 0
        if len(quarters) >= 4:
            recent = sum(ai_analysis['quarterly_trends'][q]['total'] for q in quarters[-2:])
            previous = sum(ai_analysis['quarterly_trends'][q]['total'] for q in quarters[-4:-2])
            recent_trend = ((recent - previous) / previous * 100) if previous > 0 else 0
        
        return {
            'Symbol': company['Symbol'],
            'Name': company['Name'],
            'Industry': company['Industry'],
            'Latest_Revenue': latest_revenue,
            'Latest_RD': latest_rd,
            'RD_to_Revenue': (latest_rd / latest_revenue * 100) if latest_revenue else 0,
            'AI_Mentions_Total': ai_analysis['total_mentions'],
            'AI_Strategic_Focus': ai_analysis['category_counts']['strategic_initiatives'],
            'AI_Early_Stage': ai_analysis['category_counts']['early_stage'],
            'AI_Infrastructure': ai_analysis['category_counts']['infrastructure'],
            'AI_Talent': ai_analysis['category_counts']['talent'],
            'AI_Recent_Trend': recent_trend,
            'AI_Readiness_Score': ai_readiness,
            'Key_Context': self._extract_key_context(ai_analysis['context_snippets']),
            'Investment_Opportunity': self._assess_opportunity(ai_readiness, recent_trend)
        }
    
    def _get_latest_metric(self, metric_data: List[Dict]) -> float:
        """Get the latest value for a metric"""
        if not metric_data:
            return 0.0
            
        latest = max(metric_data, key=lambda x: x['end'])
        return latest['val']
    
    def _analyze_ai_mentions(self, filings: List[Dict]) -> Dict:
        """Analyze AI-related mentions in filings with temporal and categorical analysis"""
        results = {
            'total_mentions': 0,
            'category_counts': {cat: 0 for cat in self.ai_keywords.keys()},
            'quarterly_trends': {},
            'context_snippets': [],
            'adoption_indicators': {
                'strategic_focus': 0,
                'early_stage': 0,
                'infrastructure_readiness': 0,
                'talent_investment': 0
            }
        }
        
        for filing in filings:
            text = self._get_filing_text(filing)
            if not text:
                continue
                
            text = text.lower()
            quarter = filing['report_date'][:7]  # YYYY-MM format
            
            if quarter not in results['quarterly_trends']:
                results['quarterly_trends'][quarter] = {
                    'total': 0,
                    'categories': {cat: 0 for cat in self.ai_keywords.keys()}
                }
            
            # Analyze each category of keywords
            for category, keywords in self.ai_keywords.items():
                for keyword in keywords:
                    matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
                    results['total_mentions'] += matches
                    results['category_counts'][category] += matches
                    results['quarterly_trends'][quarter]['total'] += matches
                    results['quarterly_trends'][quarter]['categories'][category] += matches
                    
                    # Capture context for significant mentions
                    if matches > 0:
                        # Find sentences containing keywords
                        sentences = re.split(r'[.!?]+', text)
                        for sentence in sentences:
                            if keyword in sentence:
                                results['context_snippets'].append({
                                    'date': filing['report_date'],
                                    'category': category,
                                    'keyword': keyword,
                                    'context': sentence.strip()
                                })
            
            # Update adoption indicators
            results['adoption_indicators']['strategic_focus'] += (
                results['category_counts']['strategic_initiatives'] * 2
            )
            results['adoption_indicators']['early_stage'] += (
                results['category_counts']['early_stage'] * 3
            )
            results['adoption_indicators']['infrastructure_readiness'] += (
                results['category_counts']['infrastructure']
            )
            results['adoption_indicators']['talent_investment'] += (
                results['category_counts']['talent'] * 2
            )
        
        return results

    def _get_filing_text(self, filing: Dict) -> str:
        """Get the text content of a filing"""
        try:
            # Ensure we have all required fields
            if not all(k in filing for k in ['cik', 'accession']):
                self.logger.error(f"Missing required fields in filing: {filing}")
                return ""
                
            accession_formatted = filing['accession'].replace('-', '')
            url = f"https://www.sec.gov/Archives/edgar/data/{filing['cik']}/{accession_formatted}/{filing['accession']}.txt"
            
            self.logger.debug(f"Fetching filing text from: {url}")
            
            # Use the fetcher's rate limiter and headers
            self.fetcher.rate_limiter.wait()
            response = requests.get(url, headers=self.fetcher.headers)
            
            if response.status_code == 200:
                self.fetcher.rate_limiter.success()
                # Parse the filing text
                soup = BeautifulSoup(response.text, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                    
                text = soup.get_text()
                # Clean up text
                text = ' '.join(text.split())
                self.logger.debug(f"Successfully fetched text for {filing['form']} filing dated {filing['filing_date']}")
                return text
            else:
                self.fetcher.rate_limiter.failure()
                self.logger.warning(f"Failed to fetch filing text. Status code: {response.status_code}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error getting filing text: {str(e)}")
            return ""
    
    def _calculate_ai_readiness(self, indicators: Dict, rd_ratio: float) -> float:
        """Calculate AI readiness score based on various indicators"""
        weights = {
            'strategic_focus': 0.25,
            'early_stage': 0.35,  # Higher weight for early stage indicators
            'infrastructure_readiness': 0.20,
            'talent_investment': 0.20
        }
        
        # Normalize each indicator to 0-1 scale
        max_values = {
            'strategic_focus': 50,
            'early_stage': 30,
            'infrastructure_readiness': 40,
            'talent_investment': 40
        }
        
        normalized_scores = {
            k: min(v / max_values[k], 1.0) 
            for k, v in indicators.items()
        }
        
        # Calculate weighted score
        base_score = sum(
            normalized_scores[k] * weights[k]
            for k in weights.keys()
        )
        
        # Adjust for R&D investment
        rd_bonus = min(rd_ratio / 20, 0.2)  # Cap R&D bonus at 20%
        
        return round((base_score + rd_bonus) * 100, 2)
    
    def _extract_key_context(self, snippets: List[Dict], max_snippets: int = 3) -> List[Dict]:
        """Extract the most relevant context snippets"""
        # Prioritize early stage and strategic snippets
        priority_categories = ['early_stage', 'strategic_initiatives']
        
        filtered_snippets = sorted(
            [s for s in snippets if s['category'] in priority_categories],
            key=lambda x: x['date'],
            reverse=True
        )
        
        return filtered_snippets[:max_snippets]
    
    def _assess_opportunity(self, readiness_score: float, trend: float) -> str:
        """Assess the opportunity level based on readiness and trend"""
        if readiness_score < 30 and trend > 50:
            return "High Potential - Early in AI journey with strong momentum"
        elif readiness_score < 40 and trend > 0:
            return "Promising - Starting AI initiatives"
        elif readiness_score > 70:
            return "Mature - Already advanced in AI adoption"
        elif readiness_score < 20:
            return "Early Stage - May need significant transformation"
        else:
            return "Moderate - Some AI initiatives in progress"

class IndustryAnalyzer:
    def analyze_industry_trends(self, results_df: pd.DataFrame) -> Dict:
        """Analyze trends by industry"""
        industry_stats = {}
        
        for industry in results_df['Industry'].unique():
            industry_data = results_df[results_df['Industry'] == industry]
            
            stats = {
                'company_count': len(industry_data),
                'avg_ai_readiness': industry_data['AI_Readiness_Score'].mean(),
                'median_ai_readiness': industry_data['AI_Readiness_Score'].median(),
                'avg_rd_ratio': industry_data['RD_to_Revenue'].mean(),
                'total_revenue': industry_data['Latest_Revenue'].sum(),
                'companies': industry_data['Symbol'].tolist()
            }
            
            # Identify opportunities
            high_potential = industry_data[
                (industry_data['AI_Readiness_Score'] < 40) & 
                (industry_data['AI_Recent_Trend'] > 0)
            ]
            
            stats['opportunities'] = [{
                'Symbol': row['Symbol'],
                'Name': row['Name'],
                'AI_Readiness': row['AI_Readiness_Score'],
                'Recent_Trend': row['AI_Recent_Trend'],
                'RD_Ratio': row['RD_to_Revenue'],
                'Assessment': row['Investment_Opportunity']
            } for _, row in high_potential.iterrows()]
            
            industry_stats[industry] = stats
        
        return industry_stats

    def generate_industry_report(self, stats: Dict) -> str:
        """Generate readable industry report focused on opportunities"""
        report = "\nAI Adoption Analysis Report\n"
        report += "========================\n\n"
        
        # Sort industries by number of opportunities
        sorted_industries = sorted(
            stats.items(),
            key=lambda x: len(x[1]['opportunities']),
            reverse=True
        )
        
        for industry, data in sorted_industries:
            report += f"{industry}\n{'-' * len(industry)}\n"
            report += f"Companies analyzed: {data['company_count']}\n"
            report += f"Average AI Readiness Score: {data['avg_ai_readiness']:.1f}\n"
            report += f"Average R&D to Revenue: {data['avg_rd_ratio']:.2f}%\n\n"
            
            if data['opportunities']:
                report += "Key Opportunities:\n"
                for opp in data['opportunities']:
                    report += f"* {opp['Name']} ({opp['Symbol']})\n"
                    report += f"  - AI Readiness Score: {opp['AI_Readiness']:.1f}\n"
                    report += f"  - Recent AI Momentum: {opp['Recent_Trend']:.1f}%\n"
                    report += f"  - Assessment: {opp['Assessment']}\n"
            else:
                report += "No significant opportunities identified in this sector.\n"
            
            report += "\n"
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Analyze AI adoption in companies')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--force-download', action='store_true',
                       help='Force download of new data instead of using cache')
    parser.add_argument('--output', type=str, default='ai_adoption_analysis.xlsx',
                       help='Output file path')
    parser.add_argument('--report', type=str, default='ai_adoption_report.txt',
                       help='Output report file path')
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger(args.verbose)
    cache = DataCache()
    fetcher = SECDataFetcher(logger, cache, args.force_download)
    analyzer = AIAdoptionAnalyzer(fetcher, logger)
    industry_analyzer = IndustryAnalyzer()
    
    try:
        # Run analysis
        results_df = analyzer.run_analysis()
        
        # Generate industry analysis
        industry_stats = industry_analyzer.analyze_industry_trends(results_df)
        industry_report = industry_analyzer.generate_industry_report(industry_stats)
        
        # Save results
        results_df.to_excel(args.output, index=False)
        
        # Save report
        with open(args.report, 'w') as f:
            f.write(industry_report)
        
        # Print report to console
        console = Console()
        console.print(industry_report)
        console.print(f"\n[green]Results saved to {args.output}[/green]")
        console.print(f"[green]Report saved to {args.report}[/green]")
        
    except Exception as e:
        console = Console()
        console.print(f"[red]Error during analysis: {str(e)}[/red]")

if __name__ == "__main__":
    main()