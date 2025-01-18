import json
import pickle
from pathlib import Path
import pandas as pd
from collections import defaultdict
import re
from typing import Dict, List, Optional
import logging
from bs4 import BeautifulSoup
import requests
import time

class OpportunityAnalyzer:
    def __init__(self, cache_dir: str = 'sec_cache'):
        self.cache_dir = Path(cache_dir)
        self.logger = self._setup_logger()
        
        # Define opportunity criteria
        self.opportunity_criteria = {
            'min_revenue': 10_000_000_000,  # $10B
            'min_rd_ratio': 0.05,           # 5% R&D to revenue
            'max_rd_ratio': 0.50,           # 50% R&D to revenue (to filter outliers)
        }
        
        # AI keywords for detailed analysis
        self.ai_keywords = {
            'exploratory': [
                'exploring ai', 'evaluating ai', 'ai pilot',
                'ai experiment', 'testing ai', 'ai assessment'
            ],
            'strategic': [
                'ai strategy', 'digital transformation', 'ai roadmap',
                'ai initiative', 'innovation strategy', 'strategic priority'
            ],
            'implementation': [
                'implementing ai', 'deploying ai', 'ai solution',
                'machine learning implementation', 'ai system',
                'ai platform'
            ]
        }

    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger('OpportunityAnalyzer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def analyze_opportunities(self, detailed_analysis_count: int = 10) -> pd.DataFrame:
        """Two-phase analysis"""
        print("\nPhase 1: Initial Screening")
        print("=======================")
        print("Scanning all companies in cache...")
        
        # First phase: Screen based on financial metrics
        preliminary_results = []
        fin_paths = list(self.cache_dir.glob('financials/*.json'))
        total = len(fin_paths)
        
        print(f"Found {total} companies to screen\n")
        for i, fin_path in enumerate(fin_paths, 1):
            cik = fin_path.stem
            print(f"\rScreening company {i}/{total} (CIK: {cik})", end="")
            result = self._analyze_financials(cik)
            if result:
                preliminary_results.append(result)
                
        print("\n")  # Clear the progress line
        
        if not preliminary_results:
            print("No companies found meeting initial criteria")
            return pd.DataFrame()
            
        # Convert to DataFrame and sort by preliminary score
        df = pd.DataFrame(preliminary_results)
        df = df.sort_values('preliminary_score', ascending=False)
        
        # Print preliminary findings
        self._print_preliminary_results(df)
        
        # Second phase: Detailed analysis of top candidates
        print(f"\nPhase 2: Detailed Analysis of Top {detailed_analysis_count} Companies")
        print("=" * (45 + len(str(detailed_analysis_count))))
        
        # Get top companies
        top_companies = df.head(detailed_analysis_count).copy()
        
        # Initialize AI analysis columns
        df['exploratory_mentions'] = 0
        df['strategic_mentions'] = 0
        df['implementation_mentions'] = 0
        df['total_ai_mentions'] = 0
        df['adoption_phase'] = "Not Analyzed"
        df['key_mentions'] = None
        
        for idx, company in top_companies.iterrows():
            try:
                print(f"\nAnalyzing CIK {company['cik']}...")
                ai_analysis = self._analyze_ai_adoption(company['cik'])
                
                if ai_analysis:
                    # Update with AI analysis results
                    for key, value in ai_analysis.items():
                        df.at[idx, key] = value
                else:
                    # Set default values if analysis fails
                    df.at[idx, 'adoption_phase'] = "Analysis Failed"
                    
            except Exception as e:
                print(f"Error analyzing CIK {company['cik']}: {str(e)}")
                df.at[idx, 'adoption_phase'] = "Error"
        
        # Save complete results
        df.to_csv('ai_opportunities_detailed.csv', index=False)
        print("\nDetailed results saved to ai_opportunities_detailed.csv")
        
        return df
        
        # Save complete results
        df.to_csv('ai_opportunities_detailed.csv', index=False)
        print("\nDetailed results saved to ai_opportunities_detailed.csv")
        
        return df

    def _analyze_financials(self, cik: str) -> Optional[Dict]:
        """Analyze company based on financial metrics"""
        try:
            # Load financial data
            fin_path = self.cache_dir / 'financials' / f"{cik}.json"
            with open(fin_path) as f:
                financials = json.load(f)
            
            # Get key metrics
            revenue = self._get_latest_metric(financials.get('revenue', []))
            rd_spend = self._get_latest_metric(financials.get('rd_expenses', []))
            rd_ratio = rd_spend / revenue if revenue > 0 else 0
            
            # Print metrics for significant companies
            if revenue >= self.opportunity_criteria['min_revenue']:
                print(f"\nAnalyzing CIK {cik}:")
                print(f"  Revenue: ${revenue:,.0f}")
                print(f"  R&D Spend: ${rd_spend:,.0f}")
                print(f"  R&D Ratio: {rd_ratio*100:.1f}%")
            
            # Get key metrics
            revenue = self._get_latest_metric(financials.get('revenue', []))
            rd_spend = self._get_latest_metric(financials.get('rd_expenses', []))
            rd_ratio = rd_spend / revenue if revenue > 0 else 0
            
            # Initial filtering
            if (revenue < self.opportunity_criteria['min_revenue'] or
                rd_ratio < self.opportunity_criteria['min_rd_ratio'] or
                rd_ratio > self.opportunity_criteria['max_rd_ratio']):
                return None
            
            # Calculate preliminary score (0-100)
            size_score = min(revenue / 100_000_000_000, 1.0) * 40  # Up to 40 points
            rd_score = min(rd_ratio / 0.15, 1.0) * 60  # Up to 60 points
            preliminary_score = round(size_score + rd_score, 1)
            
            return {
                'cik': cik,
                'revenue': revenue,
                'rd_spend': rd_spend,
                'rd_ratio': rd_ratio * 100,  # Convert to percentage
                'preliminary_score': preliminary_score
            }
            
        except Exception as e:
            self.logger.debug(f"Error in financial analysis: {str(e)}")
            return None

    def _analyze_ai_adoption(self, cik: str) -> Optional[Dict]:
        """Detailed AI adoption analysis for a single company"""
        try:
            print(f"\nDetailed Analysis for CIK {cik}")
            print("=" * (24 + len(str(cik))))
            
            # Load filing data
            filing_path = self.cache_dir / 'filings' / f"{cik}.pickle"
            with open(filing_path, 'rb') as f:
                filings = pickle.load(f)
            
            if not filings:
                print("  No filings found")
                return None
            
            # Analyze recent filings
            recent_filings = sorted(
                filings,
                key=lambda x: x.get('filing_date', ''),
                reverse=True
            )[:4]  # Last 4 filings
            
            print(f"\n  Found {len(recent_filings)} recent filings to analyze")
            
            ai_mentions = defaultdict(int)
            key_mentions = []
            
            for filing in recent_filings:
                print(f"\n  Processing {filing.get('form', 'Unknown')} from {filing.get('filing_date', 'Unknown')}...")
                print("  Fetching filing text from SEC EDGAR...")
                text = self._get_filing_text(filing)
                
                if text:
                    print(f"  Successfully retrieved filing text ({len(text)} characters)")
                    text = text.lower()
                    for category, keywords in self.ai_keywords.items():
                        category_mentions = 0
                        for keyword in keywords:
                            matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
                            if matches > 0:
                                category_mentions += matches
                                ai_mentions[category] += matches
                                
                                # Get context for key mentions
                                if len(key_mentions) < 3:
                                    context_pattern = r'[^.]*?' + re.escape(keyword) + r'[^.]*?[.]'
                                    contexts = re.findall(context_pattern, text)
                                    if contexts:
                                        key_mentions.append({
                                            'date': filing.get('filing_date'),
                                            'keyword': keyword,
                                            'context': contexts[0].strip()
                                        })
                        if category_mentions > 0:
                            print(f"    Found {category_mentions} {category} mentions")
                else:
                    print("  Failed to retrieve filing text")
                
                if text:
                    text = text.lower()
                    for category, keywords in self.ai_keywords.items():
                        for keyword in keywords:
                            matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
                            if matches > 0:
                                print(f"    Found {matches} mentions of '{keyword}'")
                                ai_mentions[category] += matches
                                
                                # Get context for key mentions
                                if len(key_mentions) < 3:  # Limit to top 3 mentions
                                    context_pattern = r'[^.]*?' + re.escape(keyword) + r'[^.]*?[.]'
                                    contexts = re.findall(context_pattern, text)
                                    if contexts:
                                        key_mentions.append({
                                            'date': filing.get('filing_date'),
                                            'keyword': keyword,
                                            'context': contexts[0].strip()
                                        })
            
            return {
                'exploratory_mentions': ai_mentions['exploratory'],
                'strategic_mentions': ai_mentions['strategic'],
                'implementation_mentions': ai_mentions['implementation'],
                'total_ai_mentions': sum(ai_mentions.values()),
                'adoption_phase': self._determine_phase(ai_mentions),
                'key_mentions': key_mentions
            }
            
        except Exception as e:
            print(f"Error in AI analysis: {str(e)}")
            return None

    def _get_filing_text(self, filing: Dict) -> Optional[str]:
        """Fetch filing text from SEC EDGAR"""
        try:
            accession = filing.get('accession', '')
            cik = filing.get('cik', '')
            
            if not accession or not cik:
                print("  Missing accession or CIK")
                return None
                
            accession_formatted = accession.replace('-', '')
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_formatted}/{accession}.txt"
            
            headers = {
                'User-Agent': 'Company Research Project tdeshane@gmail.com',
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'www.sec.gov'
            }
            
            print(f"  Requesting: {url}")
            response = requests.get(url, headers=headers)
            time.sleep(0.15)  # Rate limiting
            
            if response.status_code == 200:
                print(f"  Success! Status code: {response.status_code}")
                soup = BeautifulSoup(response.content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text = ' '.join(soup.get_text().split())
                print(f"  Retrieved {len(text)} characters of text")
                return text
            else:
                print(f"  Failed with status code: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  Error fetching filing: {str(e)}")
            return None

    def _get_latest_metric(self, metrics: List[Dict]) -> float:
        """Get latest value from financial metrics"""
        if not metrics:
            return 0.0
        return max(metrics, key=lambda x: x.get('end', ''))['val']

    def _determine_phase(self, mentions: Dict[str, int]) -> str:
        """Determine adoption phase based on mention patterns"""
        if sum(mentions.values()) == 0:
            return "No AI Activity"
        elif mentions['exploratory'] > mentions['implementation'] * 2:
            return "Early Exploration"
        elif mentions['strategic'] > mentions['implementation']:
            return "Strategic Planning"
        elif mentions['implementation'] > 0:
            return "Implementation"
        else:
            return "Early Stage"

    def _print_preliminary_results(self, df: pd.DataFrame):
        """Print initial screening results"""
        print(f"\nFound {len(df)} companies meeting initial criteria")
        print("\nTop 10 Companies by Preliminary Score:")
        print("=====================================")
        
        top_10 = df.head(10)
        for _, company in top_10.iterrows():
            print(f"\nCIK: {company['cik']}")
            print(f"Revenue: ${company['revenue']:,.0f}")
            print(f"R&D/Revenue: {company['rd_ratio']:.1f}%")
            print(f"Preliminary Score: {company['preliminary_score']:.1f}")

if __name__ == "__main__":
    analyzer = OpportunityAnalyzer()
    # Analyze top 10 companies in detail
    results = analyzer.analyze_opportunities(10)