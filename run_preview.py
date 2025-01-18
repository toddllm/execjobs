import sys
from pathlib import Path

def run_preview():
    try:
        # Import our analyzer
        from ai_adoption_preview import OpportunityAnalyzer
        
        print("Starting AI adoption preview analysis...")
        print("This will only analyze currently cached data\n")
        
        # Check cache directory exists
        cache_dir = Path('sec_cache')
        if not cache_dir.exists():
            print("Error: sec_cache directory not found!")
            return
            
        # Count available companies
        financial_files = list(cache_dir.glob('financials/*.json'))
        filing_files = list(cache_dir.glob('filings/*.pickle'))
        
        print(f"Found {len(financial_files)} companies with financial data")
        print(f"Found {len(filing_files)} companies with filing data")
        
        # Run analysis
        analyzer = OpportunityAnalyzer()
        analyzer.analyze_opportunities()
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("\nPlease ensure the main data collection is still running separately.")

if __name__ == "__main__":
    run_preview()