import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time

class EarlyAdoptionAnalyzer:
    def __init__(self):
        self.cache_dir = Path('sec_cache')
        self.early_stage_keywords = {
            'planning': [
                'exploring AI', 'evaluating AI', 'AI strategy',
                'planning to implement', 'beginning to implement',
                'AI roadmap', 'AI transformation'
            ],
            'infrastructure': [
                'data infrastructure', 'AI readiness',
                'modernizing systems', 'digital transformation',
                'cloud migration', 'preparing for AI'
            ],
            'talent': [
                'AI talent', 'AI expertise', 'hiring in AI',
                'building AI capabilities', 'AI skills',
                'AI training'
            ],
            'initial_projects': [
                'AI pilot', 'AI prototype', 'initial AI',
                'testing AI', 'AI experiment', 'first AI'
            ]
        }
        
        self.mature_signals = [
            'deployed AI', 'AI at scale', 'mature AI',
            'AI revenue', 'AI products', 'leading in AI'
        ]

    def get_filing_text(self, filing: Dict, cik: str) -> str:
        """Get filing text with robust error handling"""
        try:
            # Add CIK to filing dict if not present
            filing['cik'] = filing.get('cik', cik)
            
            accession_formatted = filing['accession'].replace('-', '')
            url = f"https://www.sec.gov/Archives/edgar/data/{filing['cik']}/{accession_formatted}/{filing['accession']}.txt"
            
            headers = {
                'User-Agent': 'Company Research todd.deshane@gmail.com',
                'Accept': 'application/json'
            }
            
            time.sleep(0.1)  # Rate limiting
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                # Try multiple parsing approaches
                text = ""
                try:
                    soup = BeautifulSoup(response.content, 'lxml')
                    text = soup.get_text()
                except:
                    try:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        text = soup.get_text()
                    except:
                        # Fallback to simple text extraction
                        text = response.text
                
                return ' '.join(text.split())
            return ""
            
        except Exception as e:
            print(f"Error fetching filing {filing.get('accession', 'unknown')}: {str(e)}")
            return ""

    def analyze_adoption_timeline(self, filings: List[Dict], company_info: Dict) -> Dict:
        """Analyze AI adoption timeline from filings"""
        timeline = {
            'early_signals': [],
            'recent_mentions': 0,
            'older_mentions': 0,
            'maturity_signals': [],
            'revenue': company_info.get('recent_revenue', 0),
            'trajectory': 'unknown'
        }
        
        try:
            # Sort filings by date
            sorted_filings = sorted(
                filings, 
                key=lambda x: x.get('filing_date', ''),
                reverse=True
            )
            
            recent_date = datetime.now()
            cutoff_date = datetime(2023, 1, 1)  # Compare recent vs older mentions
            
            for filing in sorted_filings[:5]:  # Look at 5 most recent filings
                try:
                    filing_date = datetime.strptime(filing['filing_date'], '%Y-%m-%d')
                    text = self.get_filing_text(filing, company_info['cik']).lower()
                    
                    if not text:
                        continue
                    
                    # Check for early adoption signals
                    for category, keywords in self.early_stage_keywords.items():
                        for keyword in keywords:
                            if keyword in text:
                                context = self.get_context(text, keyword)
                                if context:
                                    timeline['early_signals'].append({
                                        'date': filing['filing_date'],
                                        'category': category,
                                        'signal': keyword,
                                        'context': context
                                    })
                    
                    # Check for maturity signals
                    for signal in self.mature_signals:
                        if signal in text:
                            context = self.get_context(text, signal)
                            if context:
                                timeline['maturity_signals'].append({
                                    'date': filing['filing_date'],
                                    'signal': signal,
                                    'context': context
                                })
                    
                    # Count mentions by time period
                    if filing_date > cutoff_date:
                        timeline['recent_mentions'] += 1
                    else:
                        timeline['older_mentions'] += 1
                        
                except Exception as e:
                    print(f"Error processing filing dated {filing.get('filing_date', 'unknown')}: {str(e)}")
                    continue
            
            # Determine trajectory
            if timeline['recent_mentions'] > timeline['older_mentions'] * 1.5:
                timeline['trajectory'] = 'accelerating'
            elif timeline['recent_mentions'] < timeline['older_mentions'] * 0.5:
                timeline['trajectory'] = 'decelerating'
            else:
                timeline['trajectory'] = 'steady'
                
        except Exception as e:
            print(f"Error analyzing timeline for CIK {company_info['cik']}: {str(e)}")
        
        return timeline

    def get_context(self, text: str, keyword: str, context_chars: int = 100) -> str:
        """Extract context around keyword mention"""
        try:
            match = re.search(f'.{{0,{context_chars}}}{re.escape(keyword)}.{{0,{context_chars}}}', text)
            if match:
                return f"...{match.group(0)}..."
        except Exception as e:
            print(f"Error getting context for keyword '{keyword}': {str(e)}")
        return ""

    def analyze_companies(self):
        """Analyze all available companies"""
        results = []
        
        # Get all companies with both financial and filing data
        for filing_path in self.cache_dir.glob('financials/*.json'):
            cik = filing_path.stem
            if not (self.cache_dir / 'filings' / f"{cik}.pickle").exists():
                continue
                
            print(f"\nAnalyzing CIK: {cik}")
            
            try:
                # Load data
                with open(filing_path, 'r') as f:
                    financial_data = json.load(f)
                with open(self.cache_dir / 'filings' / f"{cik}.pickle", 'rb') as f:
                    filings = pickle.load(f)
                
                # Get recent revenue
                recent_revenue = 0
                if financial_data.get('revenue'):
                    revenues = sorted(
                        financial_data['revenue'],
                        key=lambda x: x.get('end', ''),
                        reverse=True
                    )
                    if revenues:
                        recent_revenue = revenues[0].get('val', 0)
                
                company_info = {
                    'cik': cik,
                    'recent_revenue': recent_revenue
                }
                
                # Analyze adoption timeline
                timeline = self.analyze_adoption_timeline(filings, company_info)
                
                # Score early adoption potential
                early_adoption_score = self.score_early_adoption_potential(timeline)
                
                results.append({
                    'cik': cik,
                    'revenue': recent_revenue,
                    'early_signals': len(timeline['early_signals']),
                    'maturity_signals': len(timeline['maturity_signals']),
                    'trajectory': timeline['trajectory'],
                    'early_adoption_score': early_adoption_score,
                    'recent_signals': timeline['early_signals'][:3]  # Most recent signals
                })
                
            except Exception as e:
                print(f"Error processing company {cik}: {str(e)}")
                continue
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('early_adoption_score', ascending=False)
        
        return df

    def score_early_adoption_potential(self, timeline: Dict) -> float:
        """Score company's potential as early AI adopter"""
        try:
            score = 0
            
            # More early signals is good
            score += len(timeline['early_signals']) * 2
            
            # Too many maturity signals is bad
            score -= len(timeline['maturity_signals']) * 3
            
            # Accelerating trajectory is good
            if timeline['trajectory'] == 'accelerating':
                score += 5
            elif timeline['trajectory'] == 'steady':
                score += 2
            
            # Revenue indicates resources (log scale to not overshadow other factors)
            if timeline['revenue'] > 0:
                score += min(5, np.log10(timeline['revenue'] / 1e9))
            
            return max(0, score)  # No negative scores
            
        except Exception as e:
            print(f"Error calculating score: {str(e)}")
            return 0.0

def main():
    analyzer = EarlyAdoptionAnalyzer()
    print("Starting analysis of early AI adoption patterns...")
    
    try:
        results_df = analyzer.analyze_companies()
        
        if results_df.empty:
            print("\nNo results found. Check if data files are present and accessible.")
            return
            
        print("\nTop Early AI Adoption Candidates:")
        print("--------------------------------")
        
        top_candidates = results_df.head(5)
        for _, row in top_candidates.iterrows():
            print(f"\nCompany CIK: {row['cik']}")
            print(f"Revenue: ${row['revenue']:,.0f}")
            print(f"Early Adoption Score: {row['early_adoption_score']:.1f}")
            print(f"Early Signals: {row['early_signals']}")
            print(f"Maturity Signals: {row['maturity_signals']}")
            print(f"Trajectory: {row['trajectory']}")
            if isinstance(row['recent_signals'], list) and row['recent_signals']:
                print("\nRecent Signals:")
                for signal in row['recent_signals']:
                    print(f"- {signal['date']} ({signal['category']}): {signal['signal']}")
                    print(f"  Context: {signal['context']}")
        
        # Save full results
        results_df.to_csv('early_adoption_analysis.csv', index=False)
        print("\nFull results saved to early_adoption_analysis.csv")
        
    except Exception as e:
        print(f"Error in main analysis: {str(e)}")

if __name__ == "__main__":
    main()