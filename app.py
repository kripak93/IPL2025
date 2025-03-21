from flask import Flask
import os
import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# Initialize Flask app
server = Flask(__name__)

# Load CSV data




# Define color scheme
COLORS = {
    'primary': '#1E88E5',       # Blue
    'secondary': '#FFC107',     # Amber
    'background': '#F5F7FA',    # Light gray
    'text': '#333333',          # Dark gray
    'accent': '#4CAF50',        # Green
    'warning': '#FF5722',       # Deep orange
    'chart': ['#1E88E5', '#FFC107', '#4CAF50', '#FF5722', '#9C27B0', '#E91E63'] # Chart colors
}

# Initialize Dash app 1 (Bowler Dashboard)
app = dash.Dash(
    __name__, 
    server=server, 
    url_base_pathname="/bowler-dashboard/",
    external_stylesheets=[
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css'
    ]
)

# Initialize Dash app 2 (Batsman Dashboard)
app2 = dash.Dash(
    __name__, 
    server=server, 
    url_base_pathname="/Batsman-dashboard/",
    external_stylesheets=[
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css'
    ]
)

try:
        file_path = 'C:/Users/kripa/Desktop/BallByBall2023(in).csv'
        df = pd.read_csv(file_path)
        
        # Convert BatsManName and BowlerName to string to handle mixed types
        df['BatsManName'] = df['BatsManName'].astype(str)
        df['BowlerName'] = df['BowlerName'].astype(str)
        
                # Extract unique Batsman and bowler names from the dataset (safely)
        Batsman_names = sorted([str(name) for name in df['BatsManName'].unique() if name is not None and name != 'nan'])
        bowler_names = sorted([str(name) for name in df['BowlerName'].unique() if name is not None and name != 'nan'])  

        
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()


def analyze_post_dot_ball_response(df):
    """
    Analyze how batsmen respond when facing a ball after two consecutive dot balls,
    using commentary text to identify dot balls.

    Args:
    df: DataFrame containing cricket match data with NewCommentry, BatsManName, 
        ActualRuns, BallNo, ShotType, XLanding, and YLanding columns.

    Returns:
    Tuple of DataFrames:
    - Aggregated analysis of runs scored, shot type, and response metrics.
    - Raw coordinates and detailed shot data for individual balls after consecutive dots.
    """

    # Create a copy to avoid modifying original
    analysis_df = df.copy()

    # Ensure ActualRuns is numeric
    analysis_df['ActualRuns'] = pd.to_numeric(analysis_df['ActualRuns'], errors='coerce')

    # Create dot ball flag based on commentary
    analysis_df['is_dot_ball'] = analysis_df['NewCommentry'].str.contains('DOT', case=True, na=False).astype(int)

    # Sort by match, innings, over, and ball number to maintain chronological order
    if all(col in analysis_df.columns for col in ['MatchID', 'InningsNo', 'OverNo', 'BallNo']):
        analysis_df = analysis_df.sort_values(['MatchID', 'InningsNo', 'OverNo', 'BallNo'])
    else:
        # Fallback to just BallNo if other columns don't exist
        analysis_df = analysis_df.sort_values(['BallNo'])

    # Create rolling window to identify consecutive dot balls
    analysis_df['prev_dot'] = analysis_df.groupby(['MatchID', 'InningsNo'] if 'MatchID' in analysis_df.columns and 'InningsNo' in analysis_df.columns else [])['is_dot_ball'].shift(1)
    analysis_df['prev_prev_dot'] = analysis_df.groupby(['MatchID', 'InningsNo'] if 'MatchID' in analysis_df.columns and 'InningsNo' in analysis_df.columns else [])['is_dot_ball'].shift(2)

    # Identify where two consecutive dot balls occurred
    analysis_df['after_consecutive_dots'] = (
        (analysis_df['prev_dot'] == 1) & 
        (analysis_df['prev_prev_dot'] == 1)
    )

    # Get the results for balls after consecutive dots
    results = analysis_df[analysis_df['after_consecutive_dots']].copy()

    # If no results found, return empty DataFrames
    if len(results) == 0:
        empty_aggregated = pd.DataFrame(columns=[
            'BatsManName', 'INSTANCES', 'TOTAL_RUNS', 'AVG_RUNS',
            'MOST_COMMON_SHOT', 'STRIKE_RATE', 'SHOT_BREAKDOWN', 'RUNS_BREAKDOWN',
            'TOTAL_CONSECUTIVE_DOT_INSTANCES', 'PERCENTAGE_OF_TOTAL_INSTANCES'
        ])
        empty_raw = pd.DataFrame(columns=['BatsManName', 'XLanding', 'YLanding', 'ActualRuns', 'ShotType'])
        return empty_aggregated, empty_raw

    # Function to safely get most common shot type
    def get_most_common_shot(x):
        counts = x.value_counts()
        return counts.index[0] if len(counts) > 0 else 'NA'
    
    # Extract unique wicket types only for rows where IsWicket == 1
    wicket_types = results[results['IsWicket'] == 1].groupby('BatsManName')['WicketType'] \
    .apply(lambda x: list(x.unique())).reset_index()

    wicket_types.columns = ['BatsManName', 'WICKET_TYPES']

    # Aggregations
    instances = results.groupby('BatsManName')['after_consecutive_dots'].count()
    total_runs = results.groupby('BatsManName')['ActualRuns'].sum()
    # Calculate the total wickets
    total_wickets = results.groupby('BatsManName')['IsWicket'].sum()
    avg_runs = results.groupby('BatsManName')['ActualRuns'].mean()
    most_common_shots = results.groupby('BatsManName')['ShotType'].agg(get_most_common_shot)

    

    # Create final aggregated DataFrame
    aggregated_results = pd.DataFrame({
        'INSTANCES': instances,
        'TOTAL_RUNS': total_runs,
        'AVG_RUNS': avg_runs,
        'MOST_COMMON_SHOT': most_common_shots,
        'Total_Wickets' : total_wickets
    }).reset_index()

    # Calculate strike rate (runs per ball * 100)
    aggregated_results['STRIKE_RATE'] = (aggregated_results['AVG_RUNS'] * 100).round(2)
    aggregated_results['AVG_RUNS'] = aggregated_results['AVG_RUNS'].round(2)
    

    # Merge with wicket types
    aggregated_results = aggregated_results.merge(wicket_types, on='BatsManName', how='left')

    # Safely convert WICKET_TYPES to a comma-separated string
    aggregated_results['WICKET_TYPES'] = aggregated_results['WICKET_TYPES'].apply(
    lambda x: ', '.join(map(str, x)) if isinstance(x, list) else ''
)

    # Add detailed shot breakdown
    def get_shot_breakdown(group):
        shot_counts = group['ShotType'].value_counts()
        return {shot: int(count) for shot, count in shot_counts.items()} if not shot_counts.empty else {'NA': 0}

    # Add detailed runs breakdown
    def get_runs_breakdown(group):
        runs_counts = group['ActualRuns'].value_counts()
        return {str(int(runs)): int(count) for runs, count in runs_counts.items()} if not runs_counts.empty else {'0': 0}

    shot_breakdown = results.groupby('BatsManName').apply(get_shot_breakdown).reset_index()
    shot_breakdown.columns = ['BatsManName', 'SHOT_BREAKDOWN']

    runs_breakdown = results.groupby('BatsManName').apply(get_runs_breakdown).reset_index()
    runs_breakdown.columns = ['BatsManName', 'RUNS_BREAKDOWN']

    # Convert dictionary to string for display in table
    shot_breakdown['SHOT_BREAKDOWN'] = shot_breakdown['SHOT_BREAKDOWN'].apply(lambda x: json.dumps(x))
    runs_breakdown['RUNS_BREAKDOWN'] = runs_breakdown['RUNS_BREAKDOWN'].apply(lambda x: json.dumps(x))

    # Merge breakdowns with final results
    aggregated_results = aggregated_results.merge(shot_breakdown, on='BatsManName')
    aggregated_results = aggregated_results.merge(runs_breakdown, on='BatsManName')

    # Add total dot ball statistics
    total_consecutive_dots = len(results)
    aggregated_results['TOTAL_CONSECUTIVE_DOT_INSTANCES'] = total_consecutive_dots
    aggregated_results['PERCENTAGE_OF_TOTAL_INSTANCES'] = (
        (aggregated_results['INSTANCES'] / total_consecutive_dots * 100).round(2))

    # Create raw coordinates DataFrame
    # Drop rows where 'BatType' is NaN
    results1 = results.dropna(subset=['BatType'])
    # Correct conversion to integer type
    results1['IsWicket'] = results1['IsWicket'].astype('int')

    

    raw_coordinates = results1[['BatsManName', 'XLanding', 'YLanding', 'ActualRuns','BatType', 'ShotType','IsWicket']].copy()
    raw_coordinates_with_wickets = results[results['IsWicket'] == 1].copy()
    

    return aggregated_results.sort_values('INSTANCES', ascending=False), raw_coordinates, raw_coordinates_with_wickets




# Extract unique Batsman names from the dataset (safely)
player_names = sorted([str(name) for name in df['BatsManName'].unique() if name is not None and name != 'nan'])

def plot_combined_wagon_wheel(df, x_col='XLanding', y_col='YLanding', color_col='ActualRuns', 
                             title='Wagon Wheel', player_name='BatsManName', player_batting_types='BatType'):
    """
    Creates a combined cricket wagon wheel visualization with shots colored by a specified category.
    Coordinates originate from the keeper's position with proper cricket field orientation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the shot coordinates and run outcomes
    x_col : str, default='XLanding'
        Column name for x-coordinates
    y_col : str, default='YLanding'
        Column name for y-coordinates
    color_col : str, default='ActualRuns'
        Column name containing the categories to color shots by
    title : str, default='Wagon Wheel'
        Title for the plot
    player_name : str, optional
        Name of the player whose shots are being visualized
    player_batting_types : str, optional
        Column name containing the batting types ('R' or 'L')
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure with the wagon wheel visualization
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    
    # Determine batting type based on player name and BatType column
    BatType = ''  # Default to right-handed if not specified
    
    if player_name and player_batting_types in df.columns:
        BatType = df[player_batting_types].iloc[0]  # Assuming all entries for the same player have the same BatType
    
    # Create figure
    fig = go.Figure()
    
    # Define key positions 
    keeper_pos = (0, 0)  # Origin at keeper's position
    batsman_pos = (0, 22)  # Batsman stands in front of the keeper (about 22 units)
    
    # Add field background (circle)
    radius = 200
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-radius, y0=-radius/2, x1=radius, y1=radius*1.5,
        line_color="black",
        fillcolor="#8cb369",
        opacity=0.5
    )
    
    # Add pitch (rectangle)
    pitch_width = 10
    pitch_length = 66  # Standard cricket pitch is 22 yards (66 feet)
    # Pitch starts at keeper and extends forward
    fig.add_shape(
        type="rect",
        xref="x", yref="y",
        x0=-pitch_width/2, y0=0, x1=pitch_width/2, y1=pitch_length,
        line_color="black",
        fillcolor="#d0a98f",
        opacity=0.8
    )
    
    # Add boundary circle - centered around the batsman, not the keeper
    boundary_radius = 200  # Slightly reduced to ensure all positions are inside the outer circle
    # fig.add_shape(
    #     type="circle",
    #     xref="x", yref="y",
    #     x0=batsman_pos[0]-boundary_radius, y0=batsman_pos[1]-boundary_radius, 
    #     x1=batsman_pos[0]+boundary_radius, y1=batsman_pos[1]+boundary_radius,
    #     line_color="white",
    #     line_width=2
    # )
    
    # Add batsman and keeper markers
    fig.add_trace(
        go.Scatter(
            x=[batsman_pos[0]], y=[batsman_pos[1]],
            mode='markers',
            marker=dict(size=10, color='#333'),
            name='Batsman'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[keeper_pos[0]], y=[keeper_pos[1]],
            mode='markers',
            marker=dict(size=10, color='#e63946'),
            name='Keeper'
        )
    )

    # Add dividing line between off and leg side and field annotations based on batsman's handedness
    if BatType == 'R':
        # For right-handed batsman, leg side is left (-x) and off side is right (+x)
        fig.add_annotation(x=-100, y=batsman_pos[1]+30, text="LEG SIDE", showarrow=False, 
                         font=dict(size=14, color="#333", family="Arial Black"))
        fig.add_annotation(x=100, y=batsman_pos[1]+30, text="OFF SIDE", showarrow=False, 
                         font=dict(size=14, color="#333", family="Arial Black"))
        
        # Right-handed batsman field positions - all within boundary radius
        fielding_positions = [
            # Behind batsman (relative to keeper)
            {"x": -40, "y": 10, "text": "Slip", "side": "Off"},
            {"x": -55, "y": 8, "text": "Gully", "side": "Off"},
            {"x": 40, "y": 5, "text": "Leg Slip", "side": "Leg"},
            
            # Square and forward positions (all relative to batsman's position)
            {"x": -110, "y": batsman_pos[1], "text": "Point", "side": "Off"},
            {"x": -150, "y": batsman_pos[1]-15, "text": "Third Man", "side": "Off"},
            {"x": -80, "y": batsman_pos[1]+50, "text": "Cover", "side": "Off"},
            {"x": -110, "y": batsman_pos[1]+80, "text": "Extra Cover", "side": "Off"},
            {"x": -40, "y": batsman_pos[1]+100, "text": "Mid Off", "side": "Off"},
            {"x": -60, "y": batsman_pos[1]+140, "text": "Long Off", "side": "Off"},
            
            {"x": 40, "y": batsman_pos[1]+100, "text": "Mid On", "side": "Leg"},
            {"x": 60, "y": batsman_pos[1]+140, "text": "Long On", "side": "Leg"},
            {"x": 80, "y": batsman_pos[1]+50, "text": "Mid Wicket", "side": "Leg"},
            {"x": 110, "y": batsman_pos[1]+80, "text": "Deep Mid Wicket", "side": "Leg"},
            {"x": 110, "y": batsman_pos[1], "text": "Square Leg", "side": "Leg"},
            {"x": 150, "y": batsman_pos[1]-15, "text": "Fine Leg", "side": "Leg"}
        ]
    elif BatType == 'L':  # Left-handed batsman
        # For left-handed batsman, leg side is right (+x) and off side is left (-x)
        fig.add_annotation(x=100, y=batsman_pos[1]+30, text="LEG SIDE", showarrow=False, 
                         font=dict(size=14, color="#333", family="Arial Black"))
        fig.add_annotation(x=-100, y=batsman_pos[1]+30, text="OFF SIDE", showarrow=False, 
                         font=dict(size=14, color="#333", family="Arial Black"))
        
        # Left-handed batsman field positions (mirrored) - all within boundary radius
        fielding_positions = [
            # Behind batsman (relative to keeper)
            {"x": 40, "y": 10, "text": "Slip", "side": "Off"},
            {"x": 55, "y": 8, "text": "Gully", "side": "Off"},
            {"x": -40, "y": 5, "text": "Leg Slip", "side": "Leg"},
            
            # Square and forward positions (all relative to batsman's position)
            {"x": 110, "y": batsman_pos[1], "text": "Point", "side": "Off"},
            {"x": 150, "y": batsman_pos[1]-15, "text": "Third Man", "side": "Off"},
            {"x": 80, "y": batsman_pos[1]+50, "text": "Cover", "side": "Off"},
            {"x": 110, "y": batsman_pos[1]+80, "text": "Extra Cover", "side": "Off"},
            {"x": 40, "y": batsman_pos[1]+100, "text": "Mid Off", "side": "Off"},
            {"x": 60, "y": batsman_pos[1]+140, "text": "Long Off", "side": "Off"},
            
            {"x": -40, "y": batsman_pos[1]+100, "text": "Mid On", "side": "Leg"},
            {"x": -60, "y": batsman_pos[1]+140, "text": "Long On", "side": "Leg"},
            {"x": -80, "y": batsman_pos[1]+50, "text": "Mid Wicket", "side": "Leg"},
            {"x": -110, "y": batsman_pos[1]+80, "text": "Deep Mid Wicket", "side": "Leg"},
            {"x": -110, "y": batsman_pos[1], "text": "Square Leg", "side": "Leg"},
            {"x": -150, "y": batsman_pos[1]-15, "text": "Fine Leg", "side": "Leg"}
        ]
    else:
        # If BatType is neither 'R' nor 'L', don't add any fielding positions
        fielding_positions = []
        # Still add a dividing line for reference
        fig.add_annotation(x=-100, y=batsman_pos[1]+30, text="LEFT SIDE", showarrow=False, 
                         font=dict(size=14, color="#333", family="Arial Black"))
        fig.add_annotation(x=100, y=batsman_pos[1]+30, text="RIGHT SIDE", showarrow=False, 
                         font=dict(size=14, color="#333", family="Arial Black"))
    
    # Add dividing line through the batsman
    fig.add_shape(
        type="line",
        x0=0, y0=batsman_pos[1], x1=0, y1=batsman_pos[1]+radius,
        line=dict(color="white", width=1, dash="dash"),
    )
    
    # Add bowler's and batter's end labels
    fig.add_annotation(x=0, y=-10, text="KEEPER'S END", showarrow=False, 
                     font=dict(size=12, color="#333", family="Arial"))
    fig.add_annotation(x=0, y=pitch_length+10, text="BOWLER'S END", showarrow=False, 
                     font=dict(size=12, color="#333", family="Arial"))
    
    # Add fielding positions
    for pos in fielding_positions:
        fig.add_annotation(
            x=pos["x"], 
            y=pos["y"], 
            text=pos["text"],
            showarrow=False,
            font=dict(size=9, color="#333"),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="#333",
            borderwidth=1,
            borderpad=2,
            opacity=0.8
        )
    
    # Clean data
    df_clean = df.dropna(subset=[x_col, y_col])
    
    if len(df_clean) > 0:
        # Since coordinates in data are already relative to batsman
        # We need to convert them to be relative to the keeper (origin)
        # by adding the batsman's position
        
        if color_col in df.columns:
            # Get unique categories and assign colors
            categories = df_clean[color_col].unique()
            
            # Plot shots by category
            for cat in categories:
                cat_df = df_clean[df_clean[color_col] == cat]
                
                # Skip if no data for this category
                if len(cat_df) == 0:
                    continue
                    
                # For each shot in this category, draw a line from batsman to landing point
                for i, row in cat_df.iterrows():
                    x_val = row[x_col]
                    y_val = row[y_col]
                    
                    # Landing point is relative to batsman, so add batsman's position
                    # to get it relative to keeper (origin)
                    x_adjusted = batsman_pos[0] + x_val
                    y_adjusted = batsman_pos[1] + y_val
                    
                    # Add line from batsman to landing point
                    fig.add_trace(
                        go.Scatter(
                            x=[batsman_pos[0], x_adjusted],
                            y=[batsman_pos[1], y_adjusted],
                            mode='lines',
                            line=dict(width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )
                
                # Add markers for landing points
                adjusted_x = cat_df[x_col] + batsman_pos[0]
                adjusted_y = cat_df[y_col] + batsman_pos[1]
                
                fig.add_trace(
                    go.Scatter(
                        x=adjusted_x,
                        y=adjusted_y,
                        mode='markers',
                        marker=dict(size=8),
                        name=str(cat),
                        hovertemplate='<b>' + str(cat) + '</b><br>' +
                                    'Runs: %{customdata}<br>' +
                                    'X: %{x:.1f}<br>' +
                                    'Y: %{y:.1f}<br>',
                        customdata=cat_df['ActualRuns'] if 'ActualRuns' in cat_df.columns else np.full(len(cat_df), ''),
                    )
                )
        else:
            # If no color column, plot all shots in the same color
            # Add lines from batsman to landing points
            for i, row in df_clean.iterrows():
                x_val = row[x_col]
                y_val = row[y_col]
                
                # Landing point is relative to batsman
                x_adjusted = batsman_pos[0] + x_val
                y_adjusted = batsman_pos[1] + y_val
                
                fig.add_trace(
                    go.Scatter(
                        x=[batsman_pos[0], x_adjusted],
                        y=[batsman_pos[1], y_adjusted],
                        mode='lines',
                        line=dict(width=1, color='rgba(100, 100, 100, 0.5)'),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
            
            # Add markers for landing points
            adjusted_x = df_clean[x_col] + batsman_pos[0]
            adjusted_y = df_clean[y_col] + batsman_pos[1]
            
            fig.add_trace(
                go.Scatter(
                    x=adjusted_x,
                    y=adjusted_y,
                    mode='markers',
                    marker=dict(size=8, color='rgba(100, 100, 100, 0.8)'),
                    name='Shots',
                    hovertemplate='Runs: %{customdata}<br>' +
                                'X: %{x:.1f}<br>' +
                                'Y: %{y:.1f}<br>',
                    customdata=df_clean['ActualRuns'] if 'ActualRuns' in df_clean.columns else np.full(len(df_clean), ''),
                )
            )
    
    # Update layout
    shot_count = len(df_clean)
    total_runs = df_clean['ActualRuns'].sum() if 'ActualRuns' in df_clean.columns else 'N/A'
    
    # Determine batting type label
    bat_type_label = "Right-handed" if BatType == 'R' else "Left-handed" if BatType == 'L' else "Unspecified"
    
    # Create title with player name if provided
    if player_name:
        title_text = f"{player_name}'s {title} ({bat_type_label})"
    else:
        title_text = f"{title} ({bat_type_label})"
    
    fig.update_layout(
        title=dict(
            text=f"{title_text}<br><sup>{shot_count} shots, {total_runs} total runs</sup>", 
            font=dict(size=16, color="#333")
        ),
        xaxis=dict(range=[-radius-50, radius+50], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-radius/2, radius*1.5], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='white',
        width=700,
        height=700,
        margin=dict(t=80, b=50),
        legend_title_text=color_col
    )
    
    return fig

# Define color scheme
COLORS = {
    'primary': '#1E88E5',       # Blue
    'secondary': '#FFC107',     # Amber
    'background': '#F5F7FA',    # Light gray
    'text': '#333333',          # Dark gray
    'accent': '#4CAF50',        # Green
    'warning': '#FF5722',       # Deep orange
    'chart': ['#1E88E5', '#FFC107', '#4CAF50', '#FF5722', '#9C27B0', '#E91E63'] # Chart colors
}

# Custom CSS
app2.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Cricket Analytics Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: "Segoe UI", Arial, sans-serif;
                background-color: ''' + COLORS['background'] + ''';
                color: ''' + COLORS['text'] + ''';
                margin: 0;
                padding: 0;
            }
            .dashboard-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                background-color: ''' + COLORS['primary'] + ''';
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .card {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .card-title {
                color: ''' + COLORS['primary'] + ''';
                border-bottom: 2px solid ''' + COLORS['secondary'] + ''';
                padding-bottom: 10px;
                margin-top: 0;
            }
            .selector-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                align-items: center;
            }
            .dash-table-container {
                margin-top: 10px;
                overflow-x: auto;
            }
            .dash-spreadsheet {
                border-radius: 8px;
                overflow: hidden;
                font-family: "Segoe UI", Arial, sans-serif;
            }
            .dash-spreadsheet-container th {
                background-color: ''' + COLORS['primary'] + ''';
                color: white;
                padding: 10px !important;
            }
            .dash-spreadsheet-container td {
                padding: 8px !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Dash layout setup
app2.layout = html.Div(className='dashboard-container', children=[
    # Header
    html.Div(className='header', children=[
        html.H1("Cricket Post-Dot Ball Response Analysis", style={'margin': '0', 'textAlign': 'center'}),
        html.P("Analyze how players respond when facing a ball after two consecutive dots", 
               style={'textAlign': 'center', 'marginTop': '10px', 'opacity': '0.8'})
    ]),

    # Control Panel Card
    html.Div(className='card', children=[
        html.H3("Control Panel", className='card-title'),
        html.Div(className='selector-container', children=[
            # View Type Selection
            html.Div(style={'flex': '1', 'minWidth': '250px'}, children=[
                html.Label("Select Analysis View:", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                dcc.RadioItems(
                    id='view-type',
                    options=[
                        {'label': html.Span([html.I(className="fas fa-users", style={'marginRight': '5px'}), 'All Players']), 'value': 'all'},
                        {'label': html.Span([html.I(className="fas fa-user", style={'marginRight': '5px'}), 'Individual Player']), 'value': 'individual'},
                    ],
                    value='all',
                    style={'marginTop': '8px'},
                    labelStyle={'marginRight': '15px', 'display': 'inline-block'}
                ),
            ]),
            
            # Player Dropdown
            html.Div(id='player-selection-container', style={'flex': '2', 'minWidth': '250px'}, children=[
                html.Label("Select Player:", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='player-dropdown',
                    options=[{'label': player, 'value': player} for player in player_names],
                    value=player_names[0] if player_names else None,
                    style={'width': '100%'}
                )
            ]),
        ])
    ]),

    # Performance Table Card
    html.Div(className='card', children=[
        html.H3("Performance Data", className='card-title'),
        html.Div(className='dash-table-container', children=[
            dash_table.DataTable(
                id='performance-table',
                columns=[
                    {'name': 'Batsman', 'id': 'BatsManName'},
                    {'name': 'Instances', 'id': 'INSTANCES', 'type': 'numeric'},
                    {'name': 'Total Runs', 'id': 'TOTAL_RUNS', 'type': 'numeric'},
                    {'name': 'Average Runs', 'id': 'AVG_RUNS', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Strike Rate', 'id': 'STRIKE_RATE', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Most Common Shot', 'id': 'MOST_COMMON_SHOT'},
                   # {'name': 'Shot Breakdown', 'id': 'SHOT_BREAKDOWN'},
                    #{'name': 'Runs Breakdown', 'id': 'RUNS_BREAKDOWN'},
                    {'name': 'Total wickets', 'id': 'Total_Wickets', 'type': 'numeric'},
                    {'name': 'WICKET_TYPES', 'id': 'WICKET_TYPES'},
                    {'name': '% of Total Cases', 'id': 'PERCENTAGE_OF_TOTAL_INSTANCES', 'type': 'numeric', 'format': {'specifier': '.2f'}}
                ],
                style_cell={
                    'whiteSpace': 'normal',  # Allows text to wrap within cells
                    'height': 'auto',        # Adjusts the row height based on content
                    'minWidth': '120px',     # Minimum width of each column
                    'width': '150px',        # Default width of each column
                    'maxWidth': '200px',     # Maximum width to prevent overly wide columns
                    'textAlign': 'left',     # Align text to the left for better readability
                    'padding': '10px'        # Adds padding for a polished look
                },
                page_size=10,
                style_table={
                    'overflowX': 'auto',   # Ensures horizontal scrolling
                    'maxWidth': '100%'    # Keeps the table within its container
                },
                style_header={
                    'backgroundColor': COLORS['primary'],
                    'color': 'white',
                    'fontWeight': 'bold',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgba(0, 0, 0, 0.05)',
                    },
                    {
                        'if': {'column_id': 'STRIKE_RATE', 'filter_query': '{STRIKE_RATE} > 150'},
                        'backgroundColor': 'rgba(76, 175, 80, 0.2)',  # Light green
                    },
                    {
                        'if': {'column_id': 'STRIKE_RATE', 'filter_query': '{STRIKE_RATE} < 80'},
                        'backgroundColor': 'rgba(255, 87, 34, 0.2)',  # Light orange
                    }
                ],
                sort_action='native',
                filter_action='native',
            )
        ])
    ]),

    # Charts Card
    html.Div(className='card', children=[
        html.H3("Performance Visualization", className='card-title'),
        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'}, children=[
            html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                dcc.Graph(id='runs-chart', style={'height': '400px'})
            ]),
            html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                dcc.Graph(id='strike-rate-chart', style={'height': '400px'})
            ]),
        ]),
        html.Div(id='individual-charts-container', style={'marginTop': '20px'}, children=[
            dcc.Graph(id='runs-distribution-chart', style={'height': '400px'})
        ])
    ]),

    # Wagon Wheel Card (Conditional)
    html.Div(id='wagon-wheel-container', className='card', style={'display': 'none'}, children=[
        html.H3("Wagon Wheel", className='card-title'),
        dcc.Graph(id='wagon-wheel-chart', style={'height': '700px'})
    ])
])

# Callback to show/hide player dropdown and wagon wheel based on view type
@app2.callback(
    [Output('player-selection-container', 'style'),
     Output('wagon-wheel-container', 'style')],
    Input('view-type', 'value')
)
def toggle_player_dropdown_and_wagon_wheel(view_type):
    base_style = {'flex': '2', 'minWidth': '250px'}
    if view_type == 'individual':
        return base_style, {'display': 'block'}
    else:
        return {**base_style, 'display': 'none'}, {'display': 'none'}

# Callback to update table, charts, and wagon wheel
@app2.callback(
    [Output('performance-table', 'data'),
     Output('runs-chart', 'figure'),
     Output('strike-rate-chart', 'figure'),
     Output('runs-distribution-chart', 'figure'),
     Output('wagon-wheel-chart', 'figure')],
    [Input('view-type', 'value'),
     Input('player-dropdown', 'value')]
)
def update_table_and_charts(view_type, selected_player):
    if view_type == 'all':
        # Analyze all data
        aggregated_df, raw_coordinates, raw_coordinates_with_wickets = analyze_post_dot_ball_response(df)
        wagon_wheel_chart = go.Figure()  # Empty figure for all players view
    else:
        # Filter data for the selected player
        if selected_player:
            player_data = df[df['BatsManName'] == selected_player]
            aggregated_df, raw_coordinates, raw_coordinates_with_wickets = analyze_post_dot_ball_response(player_data)
            # Generate wagon wheel for individual player
            if not raw_coordinates.empty:
                raw_df_clean = raw_coordinates.dropna(subset=['XLanding', 'YLanding', 'ActualRuns'])
                wagon_wheel_chart = plot_combined_wagon_wheel(
                    raw_df_clean,
                    x_col='YLanding',
                    y_col='XLanding',
                    player_name='BatsManName',
                    player_batting_types='BatType',
                    title=f'Wagon Wheel - {selected_player}'
                )
            else:
                wagon_wheel_chart = go.Figure()  # Empty figure if no data
        else:
            # If no player selected, return empty data
            aggregated_df = pd.DataFrame(columns=[
                'BatsManName', 'INSTANCES', 'TOTAL_RUNS', 'AVG_RUNS', 'MOST_COMMON_SHOT', 
                'STRIKE_RATE', 'SHOT_BREAKDOWN', 'RUNS_BREAKDOWN',
                'TOTAL_CONSECUTIVE_DOT_INSTANCES', 'PERCENTAGE_OF_TOTAL_INSTANCES'
            ])
            wagon_wheel_chart = go.Figure()  # Empty figure

    # Convert DataFrame to dictionary for table
    table_data = aggregated_df.to_dict('records')

    # Create charts (same as before)
    if len(aggregated_df) > 0:
        # Set up common styling for plots
        plot_layout = {
            'font': {'family': '"Segoe UI", Arial, sans-serif'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {'l': 40, 'r': 20, 't': 60, 'b': 40},
            'hovermode': 'closest',
            'height': 400,
        }
        
        if view_type == 'all':
            # For all players view, take top players for clarity
            plot_data = aggregated_df.head(10)
            
            # Total Runs Chart
            runs_chart = px.bar(
                plot_data, 
                x='BatsManName', 
                y='TOTAL_RUNS',
                title='Total Runs After Consecutive Dot Balls',
                labels={'BatsManName': 'Player', 'TOTAL_RUNS': 'Total Runs'},
                color='TOTAL_RUNS',
                color_continuous_scale=px.colors.sequential.Blues,
                text='TOTAL_RUNS'
            )
            runs_chart.update_traces(texttemplate='%{text}', textposition='outside')
            runs_chart.update_layout(**plot_layout)
            
            # Strike Rate Chart
            strike_rate_chart = px.bar(
                plot_data,
                x='BatsManName', 
                y='STRIKE_RATE',
                title='Strike Rate After Consecutive Dot Balls',
                labels={'BatsManName': 'Player', 'STRIKE_RATE': 'Strike Rate'},
                color='STRIKE_RATE',
                color_continuous_scale=px.colors.sequential.Greens,
                text='STRIKE_RATE'
            )
            strike_rate_chart.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            strike_rate_chart.update_layout(**plot_layout)
            
            # Empty pie chart for all view
            runs_distribution_chart = px.pie(
                title="Runs Distribution (Select Individual Player View)",
            )
            runs_distribution_chart.update_layout(
                title={'text': 'Runs Distribution (Select Individual Player View)', 'x': 0.5, 'xanchor': 'center'},
                showlegend=False,
                annotations=[{
                    'text': 'Select Individual Player View',
                    'showarrow': False,
                    'font': {'size': 16},
                    'x': 0.5,
                    'y': 0.5
                }]
            )
            
        else:  # Individual player view
            # For individual player, focus on their performance details
            player_row = aggregated_df[aggregated_df['BatsManName'] == selected_player].iloc[0] if len(aggregated_df) > 0 else None
            
            if player_row is not None:
                # Extract shots data
                shot_breakdown = json.loads(player_row['SHOT_BREAKDOWN']) if type(player_row['SHOT_BREAKDOWN']) == str else {}
                shot_df = pd.DataFrame(list(shot_breakdown.items()), columns=['Shot', 'Count'])
                shot_df = shot_df.sort_values('Count', ascending=False)
                
                # Extract runs data
                runs_breakdown = json.loads(player_row['RUNS_BREAKDOWN']) if type(player_row['RUNS_BREAKDOWN']) == str else {}
                runs_df = pd.DataFrame(list(runs_breakdown.items()), columns=['Runs', 'Count'])
                runs_df['Runs'] = runs_df['Runs'].astype(int)
                runs_df = runs_df.sort_values('Runs')
                
                # Map runs to labels
                runs_labels = {
                    0: 'Dot Ball',
                    1: 'Single', 
                    2: 'Double',
                    3: 'Triple',
                    4: 'Boundary',
                    6: 'Six'
                }
                runs_df['Label'] = runs_df['Runs'].map(lambda x: runs_labels.get(x, f"{x} Runs"))
                
                # Shot type chart  
                runs_chart = px.bar(
                    shot_df,
                    x='Shot',
                    y='Count',
                    title=f"{selected_player}'s Shot Selection",
                    labels={'Shot': 'Shot Type', 'Count': 'Number of Times Played'},
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Blues,
                    text='Count'
                )
                runs_chart.update_traces(texttemplate='%{text}', textposition='outside')
                runs_chart.update_layout(**plot_layout)
                
                # Strike rate comparison to average
                all_player_avg = aggregated_df['STRIKE_RATE'].mean()
                player_sr = player_row['STRIKE_RATE']
                
                sr_comparison = pd.DataFrame([
                    {'Category': f"{selected_player}", 'Strike Rate': player_sr},
                    {'Category': 'All Players Avg', 'Strike Rate': all_player_avg}
                ])
                
                strike_rate_chart = px.bar(
                    sr_comparison,
                    x='Category',
                    y='Strike Rate',
                    title='Strike Rate Comparison',
                    color='Category',
                    text='Strike Rate',
                    color_discrete_map={
                        f"{selected_player}": COLORS['primary'],
                        'All Players Avg': COLORS['secondary']
                    }
                )
                strike_rate_chart.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                strike_rate_chart.update_layout(**plot_layout)
                
                # Pie chart for runs distribution
                runs_distribution_chart = px.pie(
                    runs_df,
                    values='Count',
                    names='Label',
                    title=f"{selected_player}'s Runs Distribution",
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4,
                )
                runs_distribution_chart.update_layout(
                    title={'text': f"{selected_player}'s Runs Distribution", 'x': 0.5, 'xanchor': 'center'},
                    **plot_layout
                )
                runs_distribution_chart.update_traces(textinfo='percent+label')
                
            else:
                # Empty charts if player data not found
                runs_chart = px.bar(title=f"No data available for {selected_player}")
                strike_rate_chart = px.bar(title=f"No data available for {selected_player}")
                runs_distribution_chart = px.pie(title=f"No data available for {selected_player}")
                
                for chart in [runs_chart, strike_rate_chart, runs_distribution_chart]:
                    chart.update_layout(**plot_layout)
    else:
        # Empty charts if no data
        runs_chart = px.bar(title="No data available")
        strike_rate_chart = px.bar(title="No data available")
        runs_distribution_chart = px.pie(title="No data available")
    
    return table_data, runs_chart, strike_rate_chart, runs_distribution_chart, wagon_wheel_chart




file_path = 'C:/Users/kripa/Desktop/BallByBall2023(in).csv'
df = pd.read_csv(file_path)

# Convert BatsManName and BowlerName to string to handle mixed types
df['BatsManName'] = df['BatsManName'].astype(str)
df['BowlerName'] = df['BowlerName'].astype(str)

# Ensure IsFour and IsSix columns exist
if 'IsFour' not in df.columns:
    df['IsFour'] = (df['ActualRuns'] == 4).astype(int)

if 'IsSix' not in df.columns:
    df['IsSix'] = (df['ActualRuns'] == 6).astype(int)

    # Load CSV data


# Rename columns
df = df.rename(columns={'Xpitch_meters': 'Ypitch_meters', 'Ypitch_meters': 'Xpitch_meters'}, inplace=False)

# Normalize OverName values (strip spaces, lowercase)
df['OverName'] = df['OverName'].str.strip().str.lower()

# Define a mapping to correct spelling errors
corrections = {
    'frteen': 'fourteen',
    'siteen': 'sixteen',
    'svteen': 'seventeen',
    'egteen': 'eighteen',
    'niteen': 'nineteen',
    'twon': 'two',
}

# Define irrelevant entries to drop
irrelevant_values = ['twon', 'twtw', 'twtr', 'twfr', 'twfv', 'twsx', 
                    'twsv', 'tweg', 'twnn', 'tty', 'ttn', 'ttw', 
                    'ttt', 'ttfr', 'ttfv']

# Fix spelling errors
df['OverName'] = df['OverName'].replace(corrections)

# Remove irrelevant rows
df = df[~df['OverName'].isin(irrelevant_values)]

# Remove NaN values
df.dropna(subset=['OverName'], inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)



# Function to analyze post-boundary bowler response
def analyze_post_boundary_response(df, grouping_columns=None):
    """
    Analyze how bowlers respond when bowling a ball after conceding a boundary (four or six),
    with flexible grouping options.
    
    Args:
    df: DataFrame containing cricket match data with relevant columns.
    grouping_columns: List of column names to group by. 
                      Defaults to ['BowlerName', 'BowlTypeName'] if None.
    
    Returns:
    Tuple of (aggregated results DataFrame, filtered raw results DataFrame)
    """
    
    # Create a copy to avoid modifying original
    analysis_df = df.copy()
    
    # Ensure numeric types
    analysis_df['ActualRuns'] = pd.to_numeric(analysis_df['ActualRuns'], errors='coerce')
    
    # Default grouping if not provided
    if grouping_columns is None:
        grouping_columns = ['BowlerName', 'BowlTypeName', 'OverName']  # Include OverName and BowlTypeName
    
    # Ensure all specified grouping columns exist
    for col in grouping_columns:
        if col not in analysis_df.columns:
            raise ValueError(f"Grouping column '{col}' not found in DataFrame")
    
    # Ensure BowlTypeName is present if not in grouping columns
    if 'BowlTypeName' not in analysis_df.columns:
        analysis_df['BowlTypeName'] = 'Unknown'
    
    # Create boundary flag (four or six)
    analysis_df['is_boundary'] = ((analysis_df['IsFour'] == 1) | (analysis_df['IsSix'] == 1)).astype(int)
    
    # Sort by match, innings, over, and ball number to maintain chronological order
    if all(col in analysis_df.columns for col in ['MatchID', 'InningsNo', 'OverNo', 'BallNo']):
        analysis_df = analysis_df.sort_values(['MatchID', 'InningsNo', 'OverNo', 'BallNo'])
    else:
        analysis_df = analysis_df.sort_values(['BallNo'])
    
    # Identify balls after a boundary
    analysis_df['prev_boundary'] = analysis_df.groupby(['MatchID', 'InningsNo'])['is_boundary'].shift(1)
    analysis_df['after_boundary'] = (analysis_df['prev_boundary'] == 1)
    
    # Filter for balls bowled after boundaries
    results = analysis_df[analysis_df['after_boundary']].copy()

   
    
    if results.empty:
        return pd.DataFrame(columns=grouping_columns + [
            'INSTANCES', 'TOTAL_RUNS_CONCEDED', 'AVG_RUNS_CONCEDED',
            'DOT_BALL_PERCENTAGE', 'BOUNDARY_PERCENTAGE', 'WICKET_PERCENTAGE',
            'ECONOMY_RATE', 'RUNS_BREAKDOWN', 'WICKETS', 'TOTAL_BOUNDARY_INSTANCES',
            'PERCENTAGE_OF_TOTAL_INSTANCES'
        ]), pd.DataFrame()
    
    
    # Perform aggregations based on provided grouping columns
    # Instances count
    instances = results.groupby(grouping_columns)['after_boundary'].count()
    
    # Runs and performance metrics
    total_runs = results.groupby(grouping_columns)['ActualRuns'].sum()
    avg_runs = results.groupby(grouping_columns)['ActualRuns'].mean()
    dot_balls = results.groupby(grouping_columns).apply(lambda x: (x['ActualRuns'] == 0).sum() / len(x) * 100)
    boundary_percentage = results.groupby(grouping_columns)['is_boundary'].mean() * 100
    wicket_percentage = results.groupby(grouping_columns)['IsWicket'].mean() * 100
    wickets = results.groupby(grouping_columns)['IsWicket'].sum()
    
    # Create final result DataFrame
    final_results = pd.DataFrame({
        'INSTANCES': instances,
        'TOTAL_RUNS_CONCEDED': total_runs,
        'AVG_RUNS_CONCEDED': avg_runs,
        'DOT_BALL_PERCENTAGE': dot_balls,
        'BOUNDARY_PERCENTAGE': boundary_percentage,
        'WICKET_PERCENTAGE': wicket_percentage,
        'WICKETS': wickets
    }).reset_index()
    
    # Round numeric columns
    numeric_columns = [
        'AVG_RUNS_CONCEDED', 'DOT_BALL_PERCENTAGE', 
        'BOUNDARY_PERCENTAGE', 'WICKET_PERCENTAGE'
    ]
    for col in numeric_columns:
        final_results[col] = final_results[col].round(2)
    
    # Calculate economy rate
    final_results['ECONOMY_RATE'] = (final_results['AVG_RUNS_CONCEDED'] * 6).round(2)
    
    # Add detailed runs breakdown
    def get_runs_breakdown(group):
        runs_counts = group['ActualRuns'].value_counts()
        return {str(int(runs)): int(count) for runs, count in runs_counts.items()} if not runs_counts.empty else {'0': 0}
  
    runs_breakdown = results.groupby(grouping_columns).apply(get_runs_breakdown).reset_index()
    runs_breakdown.columns = list(grouping_columns) + ['RUNS_BREAKDOWN']
    
    # Convert dictionary to string for display in table
    runs_breakdown['RUNS_BREAKDOWN'] = runs_breakdown['RUNS_BREAKDOWN'].apply(lambda x: json.dumps(x))
    
    # Merge breakdowns with final results
    final_results = final_results.merge(runs_breakdown, on=grouping_columns)
    
    # Add total boundary statistics
    total_boundary_instances = len(results)
    final_results['TOTAL_BOUNDARY_INSTANCES'] = total_boundary_instances
    final_results['PERCENTAGE_OF_TOTAL_INSTANCES'] = (
        (final_results['INSTANCES'] / total_boundary_instances * 100).round(2)
    )
    
    return final_results.sort_values('INSTANCES', ascending=False), results




    

    
# If Xpitch and Ypitch don't exist, create them with placeholder values for visualization
if 'Xpitch_meters' not in df.columns:
    # Creating random X values between -1.5 and 1.5 (typical pitch width in meters)
    import numpy as np
    df['Xpitch_meters'] = np.random.uniform(-1.5, 1.5, len(df))

if 'Ypitch_meters' not in df.columns:
    # Creating random Y values between 0 and 20 (typical pitch length in meters)
    df['Ypitch_meters'] = np.random.uniform(0, 20, len(df))
    
# except Exception as e:
#     print(f"Error loading data: {e}")
#     # Create a sample dataframe if file cannot be loaded
#     df = pd.DataFrame()

# Extract unique bowler names from the dataset (safely)
bowler_names = sorted([str(name) for name in df['BowlerName'].unique() if name is not None and name != 'nan'])

# Define color scheme
COLORS = {
    'primary': '#1E88E5',       # Blue
    'secondary': '#FFC107',     # Amber
    'background': '#F5F7FA',    # Light gray
    'text': '#333333',          # Dark gray
    'accent': '#4CAF50',        # Green
    'warning': '#FF5722',       # Deep orange
    'chart': ['#1E88E5', '#FFC107', '#4CAF50', '#FF5722', '#9C27B0', '#E91E63'] # Chart colors
}

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Bowler Post-Boundary Analytics</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: "Segoe UI", Arial, sans-serif;
                background-color: ''' + COLORS['background'] + ''';
                color: ''' + COLORS['text'] + ''';
                margin: 0;
                padding: 0;
            }
            .dashboard-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                background-color: ''' + COLORS['primary'] + ''';
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .card {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .card-title {
                color: ''' + COLORS['primary'] + ''';
                border-bottom: 2px solid ''' + COLORS['secondary'] + ''';
                padding-bottom: 10px;
                margin-top: 0;
            }
            .selector-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                align-items: center;
            }
            .dash-table-container {
                margin-top: 10px;
                overflow-x: auto;
            }
            .dash-spreadsheet {
                border-radius: 8px;
                overflow: hidden;
                font-family: "Segoe UI", Arial, sans-serif;
            }
            .dash-spreadsheet-container th {
                background-color: ''' + COLORS['primary'] + ''';
                color: white;
                padding: 10px !important;
            }
            .dash-spreadsheet-container td {
                padding: 8px !important;
            }
            .pitch-viz {
                margin-top: 20px;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def create_3d_pitch_map(df, title="3D Pitch Map - Line and Length"):
    """
    Create a 3D pitch map to visualize the line and length of deliveries.
    
    Args:
        df: DataFrame containing ball-by-ball data with 'Xpitch', 'Ypitch' columns.
              Optionally 'ZPitch' for ball height and 'ActualRuns' for coloring.
        title: Title of the pitch map.
    
    Returns:
        A Plotly figure representing the 3D pitch map.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Create 3D figure
    fig = make_subplots(specs=[[{"type": "scene"}]])
    
    # Pitch dimensions
    pitch_length = 20  # 22 yards in meters
    pitch_width = 3.05    # 10 feet in meters
    pitch_height = 0.1    # Thickness of the pitch surface
    
    # Create pitch surface
    x_pitch = np.linspace(-pitch_width/2, pitch_width/2, 10)
    y_pitch = np.linspace(0, pitch_length, 20)
    X, Y = np.meshgrid(x_pitch, y_pitch)
    Z = np.zeros_like(X)  # Flat surface
    
    # Add pitch surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, '#D2B48C'], [1, '#C19A6B']],  # Light to darker brown
        showscale=False,
        opacity=0.9,
        name="Pitch Surface"
    ))

    
    
    # Add crease lines
    # Batting crease
    x_crease = np.linspace(-pitch_width/2, pitch_width/2, 20)
    y_crease = np.ones(20) * 1.22
    z_crease = np.ones(20) * 0.01  # Slightly above pitch surface
    fig.add_trace(go.Scatter3d(
        x=x_crease, y=y_crease, z=z_crease,
        mode='lines',
        line=dict(color='white', width=5),
        name="Batting Crease"
    ))
    
    # Bowling crease
    y_crease = np.ones(20) * 18.9
    fig.add_trace(go.Scatter3d(
        x=x_crease, y=y_crease, z=z_crease,
        mode='lines',
        line=dict(color='white', width=5),
        name="Bowling Crease"
    ))
    
    # Function to create stumps
    def add_stumps(x_center, y_pos, height=0.71, width=0.22):
        stump_positions = [x_center - width/2, x_center, x_center + width/2]
        for x_pos in stump_positions:
            # Vertical part of stump
            fig.add_trace(go.Scatter3d(
                x=[x_pos, x_pos],
                y=[y_pos, y_pos],
                z=[0, height],
                mode='lines',
                line=dict(color='black', width=8),
                showlegend=False
            ))
            # Bail on top
            if x_pos != x_center:  # Only add bails between stumps
                fig.add_trace(go.Scatter3d(
                    x=[x_pos, stump_positions[stump_positions.index(x_pos) + 
                        (1 if x_pos == stump_positions[0] else -1)]],
                    y=[y_pos, y_pos],
                    z=[height, height],
                    mode='lines',
                    line=dict(color='#8B4513', width=4),  # Brown color for bails
                    showlegend=False
                ))
    
    # Add stumps
    add_stumps(0, pitch_length)  # Batsman end
    add_stumps(0, 0)  # Bowler end
    
    # Add delivery points
    if 'Xpitch_meters' in df.columns and 'Ypitch_meters' in df.columns:
        # Default Z values if not provided
        z_values = df['ZPitch'] if 'ZPitch' in df.columns else np.ones(len(df)) * 0.2
        
        # Color by runs or default
        color_values = df['ActualRuns'] if 'ActualRuns' in df.columns else np.ones(len(df))
        
        # Ball trajectory (optional enhancement)
        if 'BowlerEndX' in df.columns and 'BowlerEndY' in df.columns:
            for i in range(len(df)):
                # Draw trajectory line for each delivery
                fig.add_trace(go.Scatter3d(
                    x=[df['BowlerEndX'].iloc[i], df['Xpitch_meters'].iloc[i]],
                    y=[df['BowlerEndY'].iloc[i], df['Ypitch_meters'].iloc[i]],
                    z=[1.8, z_values.iloc[i]],  # Approximate release height
                    mode='lines',
                    line=dict(
                        color='rgba(150,150,150,0.5)',
                        width=2,
                        dash='dot'
                    ),
                    showlegend=False
                ))
        
        # Add balls as 3D markers
        fig.add_trace(go.Scatter3d(
            x=df['Xpitch_meters'],
            y=df['Ypitch_meters'],
            z=z_values,
            mode='markers',
            marker=dict(
                size=10,
                color=color_values,
                colorscale='RdYlGn_r',  # Red for high runs, green for low
                colorbar=dict(title='Runs Conceded'),
                opacity=0.8,
                symbol='circle',
            ),
            hovertemplate='<b>Runs:</b> %{marker.color}<br>' +
                         '<b>Line:</b> %{x:.2f}<br>' +
                         '<b>Length:</b> %{y:.2f}<br>' +
                         '<b>Height:</b> %{z:.2f}<extra></extra>',
            name='Deliveries'
        ))
    
    # Update layout for 3D view
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(
                title="Line (meters from middle stump)",
                range=[-pitch_width/2 - 0.5, pitch_width/2 + 0.5],
                backgroundcolor='rgba(230, 230, 230, 0.5)',
                gridcolor='white',
                showbackground=True,
            ),
            yaxis=dict(
                title="Length (meters from stumps)",
                range=[-1, pitch_length + 1],
                backgroundcolor='rgba(230, 230, 230, 0.5)',
                gridcolor='white',
                showbackground=True,
            ),
            zaxis=dict(
                title="Height (meters)",
                range=[0, 2],
                backgroundcolor='rgba(230, 230, 230, 0.5)',
                gridcolor='white',
                showbackground=True,
            ),
            aspectratio=dict(x=1, y=4, z=0.8),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2),  # Adjusted camera angle
                up=dict(x=0, y=0, z=1)
            )
        ),
        height=800,
        width=1000,
        paper_bgcolor="#F5F7FA",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    return fig



# Dash layout setup
app.layout = html.Div(className='dashboard-container', children=[
    # Header
    html.Div(className='header', children=[
        html.H1("Bowler Post-Boundary Response Analysis", style={'margin': '0', 'textAlign': 'center'}),
        html.P("Analyze how bowlers respond on the delivery after conceding a boundary (four or six)", 
               style={'textAlign': 'center', 'marginTop': '10px', 'opacity': '0.8'})
    ]),

    # Control Panel Card
html.Div(className='card', children=[
    html.H3("Control Panel", className='card-title'),
    
    html.Div(className='selector-container', children=[
        # View Type Selection
        html.Div(style={'flex': '1', 'minWidth': '250px'}, children=[
            html.Label("Select Analysis View:", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
            dcc.RadioItems(
                id='view-type',
                options=[
                    {'label': html.Span([html.I(className="fas fa-users", style={'marginRight': '5px'}), 'All Bowlers']), 'value': 'all'},
                    {'label': html.Span([html.I(className="fas fa-user", style={'marginRight': '5px'}), 'Individual Bowler']), 'value': 'individual'},
                ],
                value='all',
                style={'marginTop': '8px'},
                labelStyle={'marginRight': '15px', 'display': 'inline-block'}
            ),
        ]),
        
        # Bowler Dropdown
        html.Div(id='bowler-selection-container', style={'flex': '2', 'minWidth': '250px'}, children=[
            html.Label("Select Bowler:", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
            dcc.Dropdown(
                id='bowler-dropdown',
                options=[{'label': bowler, 'value': bowler} for bowler in bowler_names],
                value=bowler_names[0] if bowler_names else None,
                style={'width': '100%'}
            )
        ]),
        
        # OverName Dropdown Container
        html.Div(id='over-dropdown-container', style={'flex': '1', 'minWidth': '250px'}, children=[
            html.Label("Select Over:", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
            dcc.Dropdown(
                id='over-dropdown',
                options=[{'label': over, 'value': over} for over in df['OverName'].unique()],
                value=None,  # Default to no selection (show all overs)
                style={'width': '100%'}
            )
        ]),

        # BowlTypeName Dropdown Container
        html.Div(id='bowl-type-dropdown-container', style={'flex': '1', 'minWidth': '250px'}, children=[
            html.Label("Select Bowl Type:", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
            dcc.Dropdown(
                id='bowl-type-dropdown',
                options=[{'label': bowl_type, 'value': bowl_type} for bowl_type in df['BowlTypeName'].unique()],
                value=None,  # Default to no selection (show all bowl types)
                style={'width': '100%'}
            )
        ]),
    ]),  # Close selector-container
]),  # Close Control Panel Card
                
    # Performance Table Card
    html.Div(className='card', children=[
        html.H3("Performance Data", className='card-title'),
        html.Div(className='dash-table-container', children=[
            dash_table.DataTable(
                id='performance-table',
                columns=[
                    {'name': 'Bowler', 'id': 'BowlerName'},
                    {'name': 'Bowl Type', 'id': 'BowlTypeName'},
                    {'name': 'Over', 'id': 'OverName'},
                    {'name': 'Instances', 'id': 'INSTANCES', 'type': 'numeric'},
                    {'name': 'Total Runs', 'id': 'TOTAL_RUNS_CONCEDED', 'type': 'numeric'},
                    {'name': 'Average Runs', 'id': 'AVG_RUNS_CONCEDED', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Economy Rate', 'id': 'ECONOMY_RATE', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Dot Ball %', 'id': 'DOT_BALL_PERCENTAGE', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Boundary %', 'id': 'BOUNDARY_PERCENTAGE', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Wicket %', 'id': 'WICKET_PERCENTAGE', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Wickets', 'id': 'WICKETS', 'type': 'numeric'},
                    {'name': 'Runs Breakdown', 'id': 'RUNS_BREAKDOWN'},
                    {'name': '% of Total Cases', 'id': 'PERCENTAGE_OF_TOTAL_INSTANCES', 'type': 'numeric', 'format': {'specifier': '.2f'}}
                ],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '8px',
                    'fontFamily': '"Segoe UI", Arial, sans-serif',
                },
                style_header={
                    'backgroundColor': COLORS['primary'],
                    'color': 'white',
                    'fontWeight': 'bold',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgba(0, 0, 0, 0.05)',
                    },
                    {
                        'if': {'column_id': 'ECONOMY_RATE', 'filter_query': '{ECONOMY_RATE} < 6'},
                        'backgroundColor': 'rgba(76, 175, 80, 0.2)',  # Light green (good economy)
                    },
                    {
                        'if': {'column_id': 'ECONOMY_RATE', 'filter_query': '{ECONOMY_RATE} > 9'},
                        'backgroundColor': 'rgba(255, 87, 34, 0.2)',  # Light orange (poor economy)
                    },
                    {
                        'if': {'column_id': 'DOT_BALL_PERCENTAGE', 'filter_query': '{DOT_BALL_PERCENTAGE} > 50'},
                        'backgroundColor': 'rgba(76, 175, 80, 0.2)',  # Light green (good dot ball %)
                    },
                    {
                        'if': {'column_id': 'WICKET_PERCENTAGE', 'filter_query': '{WICKET_PERCENTAGE} > 20'},
                        'backgroundColor': 'rgba(76, 175, 80, 0.2)',  # Light green (good wicket %)
                    }
                ],
                sort_action='native',
                filter_action='native',
            )
        ]),
    ]),
    
    

    
    # Charts Card
    html.Div(className='card', children=[
        html.H3("Performance Visualization", className='card-title'),
        
        # Two charts in one row for larger screens, stacked for smaller screens
        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'}, children=[
            html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                dcc.Graph(id='economy-chart', style={'height': '400px'})
            ]),
            html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                dcc.Graph(id='dot-ball-chart', style={'height': '400px'})
            ]),
            html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                dcc.Graph(id='bowl_type_chart', style={'height': '400px'})
            ]),
        ]),
        
        # Additional chart for individual analysis
        html.Div(id='individual-charts-container', style={'marginTop': '20px'}, children=[
            dcc.Graph(id='runs-distribution-chart', style={'height': '400px'})
        ]),
    ]),
    
    # Pitch Map Card
    html.Div(className='card', children=[
        html.H3("Pitch Map - Line and Length", className='card-title'),
        html.P("Visualize the line and length of deliveries bowled after conceding a boundary.", 
            style={'marginBottom': '20px', 'opacity': '0.8'}),
        
        # Outcome Filter for Pitch Map
        html.Div(id='pitch-filter-container', style={'marginBottom': '15px'}, children=[
            html.Label("Filter by Outcome:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.RadioItems(
                id='pitch-outcome-filter',
                options=[
                    {'label': 'All Deliveries', 'value': 'all'},
                    {'label': 'Dot Balls', 'value': 'dot'},
                    {'label': 'Runs Scored', 'value': 'runs'},
                    {'label': 'Wickets', 'value': 'wicket'},
                    {'label': 'Boundaries', 'value': 'boundary'}
                ],
                value='all',  # Default selection
                labelStyle={'marginRight': '15px', 'display': 'inline-block'}
            )
        ]),
        
        # Pitch Map Visualization
        dcc.Graph(id='pitch-map', style={'height': '600px'})
]),
])

# Callback to show/hide bowler dropdown based on view type
@app.callback(
    Output('bowler-selection-container', 'style'),
    Input('view-type', 'value')
)
def toggle_bowler_dropdown(view_type):
    base_style = {'flex': '2', 'minWidth': '250px'}
    if view_type == 'individual':
        return base_style
    else:
        return {**base_style, 'display': 'none'}

# Callback to show/hide pitch filter based on view type
@app.callback(
    Output('pitch-filter-container', 'style'),
    Input('view-type', 'value')
)
def toggle_pitch_filter(view_type):
    if view_type == 'individual':
        return {'marginBottom': '15px', 'display': 'block'}
    else:
        return {'display': 'none'}

# Callback to show/hide individual bowler charts
@app.callback(
    Output('individual-charts-container', 'style'),
    Input('view-type', 'value')
)
def toggle_individual_charts(view_type):
    if view_type == 'individual':
        return {'display': 'block', 'marginTop': '20px'}
    else:
        return {'display': 'none'}

@app.callback(
    [Output('performance-table', 'data'),
     Output('economy-chart', 'figure'),
     Output('dot-ball-chart', 'figure'),
     Output('bowl_type_chart', 'figure'),
     Output('runs-distribution-chart', 'figure'),
     Output('pitch-map', 'figure')],
    [Input('view-type', 'value'),
     Input('bowler-dropdown', 'value'),
     Input('over-dropdown', 'value'),
     Input('bowl-type-dropdown', 'value'),
     Input('pitch-outcome-filter', 'value')]
)
def update_dashboard(view_type, selected_bowler, over_name, bowl_type, outcome_filter):
    # Start with the full dataset
    df_filtered = df

    # Filter data based on OverName
    if over_name:
        df_filtered = df_filtered[df_filtered['OverName'] == over_name]

    # Filter data based on BowlTypeName
    if bowl_type:
        df_filtered = df_filtered[df_filtered['BowlTypeName'] == bowl_type]

    # Debugging: Print the filtered data
    print(f"Filtered data (after over and bowl type filtering): {df_filtered.head()}")

    # Analyze data based on view type
    if view_type == 'all':
        # Analyze all bowlers with the applied filters
        analysis_result, detailed_results = analyze_post_boundary_response(df_filtered)
        analysis_result1, detailed_results1 = analyze_post_boundary_response(df_filtered, ['BowlerName','BowlTypeName'])
        analysis_result2, detailed_results2 = analyze_post_boundary_response(df_filtered, ['BowlerName','OverName'])
    else:
        # Filter for the selected bowler with the applied filters
        if selected_bowler:
            df_filtered = df_filtered[df_filtered['BowlerName'] == selected_bowler]
            analysis_result, detailed_results = analyze_post_boundary_response(df_filtered, grouping_columns=['BowlerName', 'BowlTypeName', 'OverName'])
            analysis_result1, detailed_results1 = analyze_post_boundary_response(df_filtered, ['BowlerName','BowlTypeName'])
            analysis_result2, detailed_results2 = analyze_post_boundary_response(df_filtered, ['BowlerName','OverName'])
        else:
            # If no bowler is selected, return an empty DataFrame
            analysis_result = pd.DataFrame()

    # Debugging: Print the final analysis result
    print(f"Analysis result: {analysis_result.head()}")

    # Convert DataFrame to dictionary for table
    table_data = analysis_result.to_dict('records')

    # Debugging: Print the table data
    print(f"Table data: {table_data}")

    # Set up common styling for plots
    plot_layout = {
        'title': {
            'font': {'size': 15, 'color': '#333'},
            'x': 0.5,  # Center align
            'xanchor': 'center'},
        'font': {'family': '"Segoe UI", Arial, sans-serif'},
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'margin': {'l': 40, 'r': 20, 't': 60, 'b': 40},
        'hovermode': 'closest',
        'height': 400,
    }

    # Initialize charts and pitch map
    economy_chart = px.bar(title="No Data Available")
    bowl_type_chart = px.bar(title="No Data Available")
    dot_ball_chart = px.bar(title="No Data Available")
    runs_distribution_chart = px.pie(title="No Data Available")
    pitch_map = create_3d_pitch_map(pd.DataFrame(), title="No Data Available")

    # Create charts if data is available
    if len(analysis_result) > 0:
        if view_type == 'all':
            # For all bowlers view, take top bowlers for clarity
            plot_data = analysis_result.head(10)

            # Economy Rate Chart
            economy_chart = px.bar(
                plot_data, 
                x='BowlerName', 
                y='ECONOMY_RATE',
                title='Economy Rate After Conceding a Boundary',
                labels={'BowlerName': 'Bowler', 'ECONOMY_RATE': 'Economy Rate'},
                color='ECONOMY_RATE',
                color_continuous_scale=px.colors.sequential.Blues_r,
                text='ECONOMY_RATE'
            )
            economy_chart.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            economy_chart.update_layout(**plot_layout)

            # Dot Ball Percentage Chart
            dot_ball_chart = px.bar(
                plot_data,
                x='BowlerName', 
                y='DOT_BALL_PERCENTAGE',
                title='Dot Ball Percentage After Conceding a Boundary',
                labels={'BowlerName': 'Bowler', 'DOT_BALL_PERCENTAGE': 'Dot Ball %'},
                color='DOT_BALL_PERCENTAGE',
                color_continuous_scale=px.colors.sequential.Greens,
                text='DOT_BALL_PERCENTAGE'
            )
            dot_ball_chart.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            dot_ball_chart.update_layout(**plot_layout)

            # Bowl Type Chart
            bowl_type_chart = px.bar(
                plot_data,
                x='BowlTypeName', 
                y='TOTAL_RUNS_CONCEDED',
                title='Runs Conceded After a Boundary by Bowl Type',
                labels={'BowlTypeName': 'Bowl Type', 'TOTAL_RUNS_CONCEDED': 'Total Runs Conceded'},
                color='TOTAL_RUNS_CONCEDED',
                color_continuous_scale=px.colors.sequential.Greens,
                text='TOTAL_RUNS_CONCEDED'
            )
            bowl_type_chart.update_traces(texttemplate='{text:.1f}', textposition='outside')
            bowl_type_chart.update_layout(**plot_layout)

            # Filter results for top bowlers
            top_bowlers = analysis_result.head(25)['BowlerName'].tolist()
            filtered_results = detailed_results[detailed_results['BowlerName'].isin(top_bowlers)]

            # Create pitch map for top bowlers
            if not filtered_results.empty and all(col in filtered_results.columns for col in ['Xpitch_meters', 'Ypitch_meters']):
                pitch_map = create_3d_pitch_map(filtered_results, title="Pitch Map - Top 25 Bowlers")
            else:
                pitch_map = create_3d_pitch_map(pd.DataFrame(), title="No Data Available")

        else:
            # Ensure a bowler is selected
            if not selected_bowler:
                return table_data, px.bar(title="No Bowler Selected"), px.bar(title="No Bowler Selected"), px.pie(title="No Bowler Selected"), create_3d_pitch_map(pd.DataFrame(), title="No Data Available")

            # Filter for selected bowler
            filtered_analysis = analysis_result[analysis_result['BowlerName'] == selected_bowler]
            filtered_results = detailed_results[detailed_results['BowlerName'] == selected_bowler]
            filtered_analysis1 = analysis_result1[analysis_result1['BowlerName'] == selected_bowler]
            filtered_analysis2 = analysis_result2[analysis_result2['BowlerName'] == selected_bowler]
            
            print(f"View type: {view_type}")
            print(f"Selected bowler: {selected_bowler}")
            print(f"Selected over: {over_name}")
            print(f"Selected bowl type: {bowl_type}")
            print(f"Filtered data (before analysis): {df_filtered.head()}")

            if len(filtered_analysis1) > 0:
                # Economy Rate Chart (single bar for selected bowler)
                economy_chart = px.bar(
                    filtered_analysis1, 
                    x='BowlTypeName', 
                    y='ECONOMY_RATE',
                    title=f'Economy After Conceding a Boundary:<br>{selected_bowler}',
                    labels={'BowlTypeName': 'Bowl Type', 'ECONOMY_RATE': 'Economy Rate'},
                    text='INSTANCES'
                )
                economy_chart.update_traces(texttemplate='Instances: %{text}', textposition='outside')
                economy_chart.update_layout(**plot_layout)

                # Dot Ball Percentage Chart (single bar for selected bowler)
                dot_ball_chart = px.bar(
                    filtered_analysis1,
                    x='BowlTypeName', 
                    y='DOT_BALL_PERCENTAGE',
                    title=f'Dot Ball % After Conceding a Boundary:<br>{selected_bowler}',
                    labels={'BowlTypeName': 'Bowl Type', 'DOT_BALL_PERCENTAGE': 'Dot Ball %'},
                    text='INSTANCES'
                )
                dot_ball_chart.update_traces(texttemplate='Instances: %{text}', textposition='outside')
                dot_ball_chart.update_layout(**plot_layout)

                # Bowl Type Chart (single bar for selected bowler)
                bowl_type_chart = px.bar(
                    filtered_analysis1,
                    x='BowlTypeName', 
                    y='TOTAL_RUNS_CONCEDED',
                    title=f'Runs Conceded After a Boundary:<br>{selected_bowler}',
                    labels={'BowlTypeName': 'Bowl Type', 'TOTAL_RUNS_CONCEDED': 'Total Runs Conceded'},
                    text='INSTANCES'
                )
                bowl_type_chart.update_traces(texttemplate='Instances: %{text}', textposition='outside')
                bowl_type_chart.update_layout(**plot_layout)

                # Runs Distribution Chart (pie chart for selected bowler)
                runs_distribution_chart = px.pie(
                    filtered_analysis2,
                    names='OverName', 
                    values='TOTAL_RUNS_CONCEDED',
                    title=f'Runs Distribution After Boundary: {selected_bowler}',
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                runs_distribution_chart.update_layout(**plot_layout)

                # Filter based on selected outcome
                if outcome_filter == 'dot':
                    filtered_results = filtered_results[filtered_results['ActualRuns'] == 0]
                elif outcome_filter == 'runs':
                    filtered_results = filtered_results[filtered_results['ActualRuns'] > 0]
                elif outcome_filter == 'wicket':
                    filtered_results = filtered_results[filtered_results['IsWicket'] == 1]
                elif outcome_filter == 'boundary':
                    filtered_results = filtered_results[(filtered_results['IsFour'] == 1) | (filtered_results['IsSix'] == 1)]

                # Pitch Map for Individual Bowler
                if not filtered_results.empty and all(col in filtered_results.columns for col in ['Xpitch_meters', 'Ypitch_meters']):
                    pitch_map = create_3d_pitch_map(filtered_results, title=f"Pitch Map - {selected_bowler}")
                else:
                    pitch_map = create_3d_pitch_map(pd.DataFrame(), title="No Data Available")

    return table_data, economy_chart, dot_ball_chart, bowl_type_chart, runs_distribution_chart, pitch_map



# Home Route
@server.route("/")
def home():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cricket Analytics</title>
        <style>
            body {{
                font-family: "Segoe UI", Arial, sans-serif;
                background-color: {COLORS['background']};
                color: {COLORS['text']};
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }}
            .container {{
                text-align: center;
                background-color: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                max-width: 600px;
            }}
            h1 {{
                color: {COLORS['primary']};
                margin-top: 0;
            }}
            p {{
                margin: 20px 0;
                line-height: 1.6;
            }}
            .btn {{
                display: inline-block;
                background-color: {COLORS['primary']};
                color: white;
                text-decoration: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-weight: bold;
                transition: background-color 0.3s;
            }}
            .btn:hover {{
                background-color: #1565C0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>IPL Prep 2025</h1>
            <p>Explore detailed analytics for bowlers and players.</p>
            <a href="/bowler-dashboard/" class="btn">Bowler Dashboard</a>
            <a href="
            /Batsman-dashboard/" class="btn">Batsman Dashboard</a>
        </div>
    </body>
    </html>
    """

# # Run Flask App
# if __name__ == "__main__":
#     server.run(debug=True, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use the port assigned by Render
    app.run(host="0.0.0.0", port=port)
