import requests
import json
from pathlib import Path
import time

def create_cik_mapping(output_file='cik_to_company.json'):
    """
    Creates a mapping of CIK numbers to company names using SEC data.
    Saves both a CIK->Name and Name->CIK mapping for flexibility.
    """
    print("Fetching company data from SEC...")
    
    # SEC requires a user agent header
    headers = {
        'User-Agent': 'Company Research Project your@email.com'  # Replace with your email
    }
    
    try:
        # Get the SEC's company tickers data
        response = requests.get(
            'https://www.sec.gov/files/company_tickers.json',
            headers=headers
        )
        response.raise_for_status()
        
        # Process the data
        company_data = response.json()
        
        # Create both forward and reverse mappings
        mapping = {
            'cik_to_name': {},
            'name_to_cik': {}
        }
        
        # Process each company
        for entry in company_data.values():
            # Pad CIK to 10 digits as often required by SEC
            cik_str = str(entry['cik_str']).zfill(10)
            name = entry['title']
            ticker = entry['ticker']
            
            # Store both the full name and ticker symbol
            mapping['cik_to_name'][cik_str] = {
                'name': name,
                'ticker': ticker
            }
            
            # Store CIK by both name and ticker for reverse lookup
            mapping['name_to_cik'][name.lower()] = cik_str
            mapping['name_to_cik'][ticker.lower()] = cik_str
        
        # Save to file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"Successfully created mapping with {len(mapping['cik_to_name'])} companies")
        print(f"Saved to {output_path.absolute()}")
        
        # Print a few examples
        print("\nExample mappings:")
        for i, (cik, info) in enumerate(list(mapping['cik_to_name'].items())[:5]):
            print(f"{cik}: {info['name']} ({info['ticker']})")
            
        return mapping
        
    except Exception as e:
        print(f"Error creating mapping: {str(e)}")
        return None

if __name__ == "__main__":
    create_cik_mapping()
