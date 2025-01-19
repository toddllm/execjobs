from flask import Flask, render_template_string
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import numpy as np

app = Flask(__name__)

# Updated HTML template with proper Plotly initialization
template = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Adoption Opportunities</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            margin-bottom: 20px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .description {
            color: #666;
            margin-bottom: 20px;
        }
        #plotly-chart {
            width: 100%;
            height: 700px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Adoption Opportunities</h1>
            <div class="description">
                <p>Visualization of companies based on AI readiness and revenue</p>
                <p><strong>Ideal Opportunities:</strong> High revenue (right side) + Low AI readiness (bottom) + Positive trend (upward triangle)</p>
                <p>• Triangle direction shows trend (↑ positive, ↓ negative)</p>
                <p>• Size indicates trend magnitude</p>
                <p>• Highlighted companies are prime candidates for executive AI leadership</p>
            </div>
        </div>
        <div id="plotly-chart"></div>
    </div>
    <script>
        var graphs = {{graphJSON | safe}};
        Plotly.newPlot('plotly-chart', graphs.data, graphs.layout);
    </script>
</body>
</html>
'''

def create_visualization():
    # Read the Excel file
    df = pd.read_excel('detailed.xlsx')
    
    # Convert revenue to billions for better display
    df['Revenue_Billions'] = df['Latest_Revenue'] / 1_000_000_000
    
    # Create the base figure
    fig = go.Figure()

    # Get unique industries and create color map
    industries = sorted(df['Industry'].unique())
    colors = px.colors.qualitative.Set3[:len(industries)]
    color_map = dict(zip(industries, colors))

    # Split data by trend direction
    df_positive = df[df['AI_Recent_Trend'] >= 0]
    df_negative = df[df['AI_Recent_Trend'] < 0]

    # Add traces for each industry (positive trends)
    for industry in industries:
        mask = (df_positive['Industry'] == industry)
        if mask.any():
            industry_data = df_positive[mask]
            
            fig.add_trace(go.Scatter(
                x=industry_data['Revenue_Billions'],
                y=industry_data['AI_Readiness_Score'],
                mode='markers',
                marker=dict(
                    size=np.clip(np.abs(industry_data['AI_Recent_Trend']) + 15, 20, 50),
                    symbol='triangle-up',
                    color=color_map.get(industry, '#808080'),
                    line=dict(width=1, color='white'),
                    opacity=0.7
                ),
                name=industry,
                text=industry_data.apply(lambda x: 
                    f"{x['Name']} ({x['Symbol']})<br>" +
                    f"Revenue: ${x['Revenue_Billions']:.1f}B<br>" +
                    f"AI Readiness: {x['AI_Readiness_Score']:.1f}<br>" +
                    f"AI Trend: +{x['AI_Recent_Trend']:.1f}%<br>" +
                    f"{x['Investment_Opportunity']}", 
                    axis=1
                ),
                hoverinfo='text'
            ))

    # Add traces for each industry (negative trends)
    for industry in industries:
        mask = (df_negative['Industry'] == industry)
        if mask.any():
            industry_data = df_negative[mask]
            
            fig.add_trace(go.Scatter(
                x=industry_data['Revenue_Billions'],
                y=industry_data['AI_Readiness_Score'],
                mode='markers',
                marker=dict(
                    size=np.clip(np.abs(industry_data['AI_Recent_Trend']) + 15, 20, 50),
                    symbol='triangle-down',
                    color=color_map.get(industry, '#808080'),
                    line=dict(width=1, color='white'),
                    opacity=0.7
                ),
                name=f"{industry} (↓)",
                showlegend=False,
                text=industry_data.apply(lambda x: 
                    f"{x['Name']} ({x['Symbol']})<br>" +
                    f"Revenue: ${x['Revenue_Billions']:.1f}B<br>" +
                    f"AI Readiness: {x['AI_Readiness_Score']:.1f}<br>" +
                    f"AI Trend: {x['AI_Recent_Trend']:.1f}%<br>" +
                    f"{x['Investment_Opportunity']}", 
                    axis=1
                ),
                hoverinfo='text'
            ))

    # Update layout
    fig.update_layout(
        title='AI Adoption Opportunities by Company',
        xaxis_title='Annual Revenue (Billions USD)',
        yaxis_title='AI Readiness Score',
        showlegend=True,
        width=1100,
        height=700,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    # Highlight prime opportunities
    opportunities = df[
        (df['Revenue_Billions'] > df['Revenue_Billions'].quantile(0.75)) &
        (df['AI_Readiness_Score'] < 40) &
        (df['AI_Recent_Trend'] > 0)
    ].sort_values('Revenue_Billions', ascending=False).head(10)

    for _, row in opportunities.iterrows():
        fig.add_annotation(
            x=row['Revenue_Billions'],
            y=row['AI_Readiness_Score'],
            text=row['Symbol'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#2563eb",
            ax=30,
            ay=-30,
            font=dict(size=12, color="#2563eb"),
            bgcolor="white",
            bordercolor="#2563eb",
            borderwidth=2
        )

    return fig

@app.route('/')
def home():
    fig = create_visualization()
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template_string(template, graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True, port=5000)