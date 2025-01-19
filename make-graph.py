import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from adjustText import adjust_text

class AIOpportunityAnalyzer:
    def __init__(self, cache_dir='sec_cache'):
        self.cache_dir = Path(cache_dir)
        # Load company mapping
        with open('cik_to_company.json', 'r') as f:
            mapping = json.load(f)
            self.cik_to_name = mapping['cik_to_name']
        
        # Expanded list of mature AI companies to exclude
        self.mature_ai_companies = {
            'GOOG', 'MSFT', 'AAPL', 'ACN', 'META', 'INTC', 
            'QCOM', 'NVDA', 'AMD', 'IBM', 'CRM', 'AMZN',
            'TSM', 'MU'
        }
        
    def load_financial_data(self):
        """Load and process financial data for all companies"""
        data = []
        
        for fin_file in self.cache_dir.glob('financials/*.json'):
            cik = fin_file.stem
            padded_cik = cik.zfill(10)
            company_info = self.cik_to_name.get(padded_cik, {})
            
            if company_info.get('ticker') in self.mature_ai_companies:
                continue
                
            with open(fin_file, 'r') as f:
                financials = json.load(f)
            
            revenue = self._get_latest_metric(financials.get('revenue', []))
            rd_spend = self._get_latest_metric(financials.get('rd_expenses', []))
            rd_ratio = (rd_spend / revenue * 100) if revenue > 0 else 0
            
            early_stage_score = (
                (min(rd_ratio, 30) / 30) * 40 +  
                (min(revenue / 50e9, 1) * 30) +   
                (min(rd_ratio * 0.5, 15))         
            )
            
            if rd_ratio > 40:
                early_stage_score *= 0.7
            
            data.append({
                'cik': cik,
                'name': company_info.get('name', f'CIK: {cik}'),
                'ticker': company_info.get('ticker', 'Unknown'),
                'revenue': revenue,
                'rd_spend': rd_spend,
                'rd_ratio': rd_ratio,
                'early_stage_score': early_stage_score
            })
        
        return pd.DataFrame(data)
    
    def _get_latest_metric(self, metrics):
        if not metrics:
            return 0.0
        return max(metrics, key=lambda x: x.get('end', ''))['val']
    
    def analyze_opportunities(self, min_revenue=10e9, min_rd_ratio=5):
        df = self.load_financial_data()
        
        df = df[
            (df['revenue'] >= min_revenue) & 
            (df['rd_ratio'] >= min_rd_ratio)
        ].copy()
        
        df = df.sort_values('early_stage_score', ascending=False)
        
        self._plot_opportunities(df)
        
        return df
    
    def _plot_opportunities(self, df):
        plt.figure(figsize=(15, 10))
        
        # Top subplot - Scatter plot
        ax1 = plt.subplot(2, 1, 1)
        
        # Create scatter plot
        scatter = ax1.scatter(
            df['rd_ratio'], 
            df['early_stage_score'],
            s=df['revenue'] / 1e9 * 5,  # Adjusted size scaling
            alpha=0.6,
            c=df['early_stage_score'],
            cmap='viridis'
        )
        
        # Create list of texts to adjust
        texts = []
        for _, row in df.iterrows():
            # Only label points with score > 20 to reduce clutter
            if row['early_stage_score'] > 20:
                t = ax1.text(
                    row['rd_ratio'], 
                    row['early_stage_score'],
                    row['ticker'],
                    fontsize=8
                )
                texts.append(t)
        
        # Adjust text positions to avoid overlap
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
            expand_points=(1.5, 1.5)
        )
        
        ax1.set_xlabel('R&D to Revenue Ratio (%)')
        ax1.set_ylabel('Early-Stage Opportunity Score')
        ax1.set_title('Early-Stage AI Adoption Opportunities\n(excluding mature AI companies)')
        
        # Add legend for bubble size
        legend_elements = [
            plt.scatter([], [], s=100, c='gray', alpha=0.6, label='$10B Revenue'),
            plt.scatter([], [], s=250, c='gray', alpha=0.6, label='$50B Revenue'),
            plt.scatter([], [], s=500, c='gray', alpha=0.6, label='$100B Revenue')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Add colorbar
        plt.colorbar(scatter, label='Opportunity Score')
        
        # Bottom subplot - Bar chart
        ax2 = plt.subplot(2, 1, 2)
        top_companies = df.head(8)
        bars = ax2.bar(
            top_companies['ticker'],
            top_companies['early_stage_score']
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2., 
                height,
                f'{height:.1f}',
                ha='center', 
                va='bottom'
            )
        
        ax2.set_title('Top Early-Stage AI Opportunities')
        ax2.set_xlabel('Company')
        ax2.set_ylabel('Opportunity Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('ai_opportunities.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print analysis
        print("\nTop Early-Stage AI Adoption Opportunities:")
        print("========================================")
        
        for _, row in df.head(8).iterrows():
            print(f"\n{row['name']} ({row['ticker']}):")
            print(f"  Revenue: ${row['revenue']/1e9:.1f}B")
            print(f"  R&D Ratio: {row['rd_ratio']:.1f}%")
            print(f"  Opportunity Score: {row['early_stage_score']:.1f}")

if __name__ == "__main__":
    analyzer = AIOpportunityAnalyzer()
    results = analyzer.analyze_opportunities()