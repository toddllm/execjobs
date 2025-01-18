import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

class QuickAnalyzer:
    def __init__(self):
        self.cache_dir = Path('sec_cache')

    def get_company_data(self) -> pd.DataFrame:
        """Get basic data about companies we have so far"""
        results = []
        
        for filing_path in self.cache_dir.glob('financials/*.json'):
            cik = filing_path.stem
            filing_pickle = self.cache_dir / 'filings' / f"{cik}.pickle"
            
            if not filing_pickle.exists():
                continue
                
            try:
                # Load financial data
                with open(filing_path, 'r') as f:
                    financial_data = json.load(f)
                
                # Get latest revenue
                revenue = 0
                if financial_data.get('revenue'):
                    revenue_entries = sorted(
                        financial_data['revenue'],
                        key=lambda x: x.get('end', ''),
                        reverse=True
                    )
                    if revenue_entries:
                        revenue = revenue_entries[0].get('val', 0)
                
                # Load filings metadata
                with open(filing_pickle, 'rb') as f:
                    filings = pickle.load(f)
                
                # Get latest filing date and convert to datetime
                filing_dates = [
                    datetime.strptime(f.get('filing_date', '2000-01-01'), '%Y-%m-%d')
                    for f in filings 
                    if f.get('filing_date')
                ]
                latest_date = max(filing_dates) if filing_dates else None
                
                # Count filing types
                filing_counts = {
                    '10-K': sum(1 for f in filings if f.get('form') == '10-K'),
                    '10-Q': sum(1 for f in filings if f.get('form') == '10-Q')
                }
                
                # Get most recent filing form
                latest_form = None
                if latest_date:
                    latest_filings = [
                        f.get('form') 
                        for f in filings 
                        if f.get('filing_date') == latest_date.strftime('%Y-%m-%d')
                    ]
                    latest_form = latest_filings[0] if latest_filings else None
                
                results.append({
                    'cik': cik,
                    'latest_revenue': revenue,
                    'latest_filing_date': latest_date,
                    'latest_form': latest_form,
                    'num_10k': filing_counts['10-K'],
                    'num_10q': filing_counts['10-Q']
                })
                
            except Exception as e:
                print(f"Error processing {cik}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        return df

def main():
    print("Analyzing currently downloaded data...")
    analyzer = QuickAnalyzer()
    df = analyzer.get_company_data()
    
    if df.empty:
        print("No data found in cache yet.")
        return
    
    print("\nCurrent Data Collection Status:")
    print(f"Companies processed so far: {len(df)}")
    
    print("\nRevenue Distribution:")
    revenue_stats = df['latest_revenue'].describe()
    print(f"Mean Revenue: ${revenue_stats['mean']:,.0f}")
    print(f"Max Revenue: ${revenue_stats['max']:,.0f}")
    print(f"Min Revenue: ${revenue_stats['min']:,.0f}")
    
    print("\nFiling Counts:")
    print(f"Total 10-Ks: {df['num_10k'].sum()}")
    print(f"Total 10-Qs: {df['num_10q'].sum()}")
    print(f"Average 10-Ks per company: {df['num_10k'].mean():.1f}")
    print(f"Average 10-Qs per company: {df['num_10q'].mean():.1f}")
    
    print("\nMost Recent Filings:")
    # Sort by latest_filing_date and handle None values
    recent_companies = df.sort_values(
        'latest_filing_date', 
        ascending=False,
        na_position='last'
    ).head(5)
    
    for _, row in recent_companies.iterrows():
        if pd.notnull(row['latest_filing_date']):
            date_str = row['latest_filing_date'].strftime('%Y-%m-%d')
            print(f"CIK {row['cik']}: {date_str} ({row['latest_form']}) - Revenue: ${row['latest_revenue']:,.0f}")
    
    print("\nLargest Companies by Revenue:")
    largest_companies = df.nlargest(5, 'latest_revenue')
    for _, row in largest_companies.iterrows():
        print(f"CIK {row['cik']}: ${row['latest_revenue']:,.0f}")
    
    # Save progress data
    df.to_csv('collection_progress.csv', index=False)
    print("\nDetailed progress saved to collection_progress.csv")

if __name__ == "__main__":
    main()