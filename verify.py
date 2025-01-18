import json
import pickle
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

def setup_logging():
    """Set up logging to console only - no file writing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    return logging.getLogger('verify')

def safe_read_json(file_path: Path) -> Dict:
    """Safely read JSON file without modifying it"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {}

def safe_read_pickle(file_path: Path) -> List:
    """Safely read pickle file without modifying it"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return []

def check_financial_file_content(data: Dict) -> Dict:
    """Analyze content of a financial data file"""
    stats = {
        'has_revenue': False,
        'has_rd': False,
        'has_operating_expenses': False,
        'has_net_income': False,
        'latest_date': None,
        'earliest_date': None,
        'data_points': 0
    }
    
    if not data:
        return stats
        
    # Check for required metrics
    metrics = ['revenue', 'rd_expenses', 'operating_expenses', 'net_income']
    for metric in metrics:
        if metric in data and data[metric]:
            stats[f'has_{metric}'] = True
            stats['data_points'] += len(data[metric])
            
            # Track date range if available
            dates = [entry.get('end') for entry in data[metric] if 'end' in entry]
            if dates:
                if not stats['latest_date'] or max(dates) > stats['latest_date']:
                    stats['latest_date'] = max(dates)
                if not stats['earliest_date'] or min(dates) < stats['earliest_date']:
                    stats['earliest_date'] = min(dates)
    
    return stats

def check_filing_content(filings: List) -> Dict:
    """Analyze content of filing data"""
    stats = {
        'total_filings': len(filings),
        'has_10k': False,
        'has_10q': False,
        'latest_filing_date': None,
        'earliest_filing_date': None,
        'form_counts': {},
        'date_gaps': False
    }
    
    if not filings:
        return stats
        
    # Analyze filings
    dates = []
    for filing in filings:
        # Count form types
        form = filing.get('form', 'UNKNOWN')
        stats['form_counts'][form] = stats['form_counts'].get(form, 0) + 1
        
        if form == '10-K':
            stats['has_10k'] = True
        elif form == '10-Q':
            stats['has_10q'] = True
            
        # Track filing dates
        if 'filing_date' in filing:
            dates.append(filing['filing_date'])
    
    # Analyze date coverage
    if dates:
        dates.sort()
        stats['latest_filing_date'] = max(dates)
        stats['earliest_filing_date'] = min(dates)
        
        # Check for unusual gaps (more than 4 months between filings)
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).days
            if gap > 120:  # More than 4 months
                stats['date_gaps'] = True
                break
    
    return stats

def check_cache_details(cache_dir: Path) -> Dict:
    """Perform detailed cache analysis"""
    stats = {
        'total_companies': 0,
        'financial_stats': {
            'companies_with_revenue': 0,
            'companies_with_rd': 0,
            'latest_data_date': None,
            'earliest_data_date': None,
            'avg_data_points': 0
        },
        'filing_stats': {
            'companies_with_10k': 0,
            'companies_with_10q': 0,
            'latest_filing_date': None,
            'earliest_filing_date': None,
            'total_filings': 0,
            'form_distribution': {}
        },
        'potential_issues': []
    }
    
    if not cache_dir.exists():
        stats['potential_issues'].append("Cache directory not found")
        return stats
        
    # Analyze financial files
    total_data_points = 0
    financial_files = list((cache_dir / 'financials').glob('*.json'))
    stats['total_companies'] = len(financial_files)
    
    for file_path in financial_files:
        data = safe_read_json(file_path)
        content_stats = check_financial_file_content(data)
        
        if content_stats['has_revenue']:
            stats['financial_stats']['companies_with_revenue'] += 1
        if content_stats['has_rd']:
            stats['financial_stats']['companies_with_rd'] += 1
            
        total_data_points += content_stats['data_points']
        
        # Track date range
        if content_stats['latest_date']:
            if (not stats['financial_stats']['latest_data_date'] or 
                content_stats['latest_date'] > stats['financial_stats']['latest_data_date']):
                stats['financial_stats']['latest_data_date'] = content_stats['latest_date']
                
        if content_stats['earliest_date']:
            if (not stats['financial_stats']['earliest_data_date'] or 
                content_stats['earliest_date'] < stats['financial_stats']['earliest_data_date']):
                stats['financial_stats']['earliest_data_date'] = content_stats['earliest_date']
    
    if stats['total_companies'] > 0:
        stats['financial_stats']['avg_data_points'] = total_data_points / stats['total_companies']
    
    # Analyze filing files
    for file_path in (cache_dir / 'filings').glob('*.pickle'):
        filings = safe_read_pickle(file_path)
        content_stats = check_filing_content(filings)
        
        stats['filing_stats']['total_filings'] += content_stats['total_filings']
        
        if content_stats['has_10k']:
            stats['filing_stats']['companies_with_10k'] += 1
        if content_stats['has_10q']:
            stats['filing_stats']['companies_with_10q'] += 1
            
        # Update form distribution
        for form, count in content_stats['form_counts'].items():
            stats['filing_stats']['form_distribution'][form] = (
                stats['filing_stats']['form_distribution'].get(form, 0) + count
            )
            
        # Track date range
        if content_stats['latest_filing_date']:
            if (not stats['filing_stats']['latest_filing_date'] or 
                content_stats['latest_filing_date'] > stats['filing_stats']['latest_filing_date']):
                stats['filing_stats']['latest_filing_date'] = content_stats['latest_filing_date']
                
        if content_stats['earliest_filing_date']:
            if (not stats['filing_stats']['earliest_filing_date'] or 
                content_stats['earliest_filing_date'] < stats['filing_stats']['earliest_filing_date']):
                stats['filing_stats']['earliest_filing_date'] = content_stats['earliest_filing_date']
                
        if content_stats['date_gaps']:
            stats['potential_issues'].append(f"Found date gaps in filings for {file_path.stem}")
    
    # Check for potential issues
    if stats['financial_stats']['companies_with_revenue'] < stats['total_companies'] * 0.8:
        stats['potential_issues'].append("Less than 80% of companies have revenue data")
    if stats['financial_stats']['companies_with_rd'] < stats['total_companies'] * 0.5:
        stats['potential_issues'].append("Less than 50% of companies have R&D data")
    if stats['filing_stats']['companies_with_10k'] < stats['total_companies'] * 0.8:
        stats['potential_issues'].append("Less than 80% of companies have 10-K filings")
    
    return stats

def check_output_details(output_file: Path) -> Dict:
    """Perform detailed output file analysis"""
    stats = {
        'exists': False,
        'row_count': 0,
        'column_coverage': {},
        'value_ranges': {},
        'potential_issues': []
    }
    
    if not output_file.exists():
        return stats
        
    stats['exists'] = True
    
    try:
        df = pd.read_excel(output_file)
        stats['row_count'] = len(df)
        
        # Check column presence and completeness
        expected_columns = {
            'Symbol', 'Name', 'Industry', 'AI_Readiness_Score', 
            'Latest_Revenue', 'Latest_RD', 'RD_to_Revenue',
            'AI_Mentions_Total', 'AI_Recent_Trend'
        }
        
        for col in expected_columns:
            if col in df.columns:
                non_null = df[col].notna().sum()
                stats['column_coverage'][col] = f"{(non_null/len(df)*100):.1f}%"
            else:
                stats['potential_issues'].append(f"Missing column: {col}")
        
        # Check value ranges for numeric columns
        numeric_cols = ['AI_Readiness_Score', 'RD_to_Revenue', 'AI_Recent_Trend']
        for col in numeric_cols:
            if col in df.columns:
                stats['value_ranges'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean()
                }
                
                # Check for suspicious values
                if col == 'AI_Readiness_Score' and df[col].max() > 100:
                    stats['potential_issues'].append(f"Found {col} > 100")
                if col == 'RD_to_Revenue' and df[col].max() > 100:
                    stats['potential_issues'].append(f"Found {col} > 100%")
        
        # Check industry distribution
        if 'Industry' in df.columns:
            industry_counts = df['Industry'].value_counts()
            stats['industry_distribution'] = industry_counts.to_dict()
            
            if len(industry_counts) < 3:
                stats['potential_issues'].append("Very few industries represented")
        
    except Exception as e:
        stats['potential_issues'].append(f"Error analyzing output: {str(e)}")
    
    return stats

def main():
    logger = setup_logging()
    cache_dir = Path('sec_cache')
    output_file = Path('ai_adoption_analysis.xlsx')
    
    logger.info("Running enhanced verification checks...")
    
    # Check cache details
    cache_stats = check_cache_details(cache_dir)
    
    logger.info(f"\nData Coverage:")
    logger.info(f"Total companies: {cache_stats['total_companies']}")
    logger.info(f"Companies with revenue data: {cache_stats['financial_stats']['companies_with_revenue']}")
    logger.info(f"Companies with R&D data: {cache_stats['financial_stats']['companies_with_rd']}")
    logger.info(f"Companies with 10-K filings: {cache_stats['filing_stats']['companies_with_10k']}")
    
    logger.info(f"\nData Timeline:")
    logger.info(f"Financial data range: {cache_stats['financial_stats']['earliest_data_date']} to {cache_stats['financial_stats']['latest_data_date']}")
    logger.info(f"Filing data range: {cache_stats['filing_stats']['earliest_filing_date']} to {cache_stats['filing_stats']['latest_filing_date']}")
    
    logger.info(f"\nFiling Distribution:")
    for form, count in cache_stats['filing_stats']['form_distribution'].items():
        logger.info(f"- {form}: {count}")
    
    # Check output details
    output_stats = check_output_details(output_file)
    if output_stats['exists']:
        logger.info(f"\nOutput Analysis:")
        logger.info(f"Companies analyzed: {output_stats['row_count']}")
        
        if output_stats['column_coverage']:
            logger.info("\nColumn Coverage:")
            for col, coverage in output_stats['column_coverage'].items():
                logger.info(f"- {col}: {coverage}")
        
        if output_stats['value_ranges']:
            logger.info("\nMetric Ranges:")
            for col, ranges in output_stats['value_ranges'].items():
                logger.info(f"- {col}: min={ranges['min']:.1f}, max={ranges['max']:.1f}, mean={ranges['mean']:.1f}")
    
    # Report potential issues
    all_issues = cache_stats['potential_issues']
    if output_stats['exists']:
        all_issues.extend(output_stats['potential_issues'])
    
    if all_issues:
        logger.info("\nPotential Issues:")
        for issue in all_issues:
            logger.warning(f"- {issue}")
    else:
        logger.info("\n✅ No major issues detected")

if __name__ == "__main__":
    main()