

from flask import Flask
import os
import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import re
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import numpy as np


# Initialize the Flask server (only once)
server = Flask(__name__)

# Initialize Dash apps with the same server
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/bowler-dashboard/",
    external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css']
)

app2 = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/batsman-dashboard/",
    external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css']
)

# app3 = dash.Dash(
#     __name__,
#     server=server,
#     url_base_pathname="/speed-dashboard/",
#     external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css', dbc.themes.BOOTSTRAP]
# )

# Define Dash app layouts
app.layout = html.Div("Hello, Bowler Dashboard!")
app2.layout = html.Div("Hello, Batsman Dashboard!")
#app3.layout = html.Div("Hello, Speed Dashboard!")

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


try:
    

    # Get the current directory of the running script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    

    # Set the current directory to your desktop
    #current_dir = os.path.expanduser("~/Desktop")
    
    # Build the relative path to the CSV file
    file_path = os.path.join(current_dir, "BallByBall2024(in).csv")
    
    # Load the CSV file
    df = pd.read_csv(file_path)    
    
    # file_path = 'C:/Users/kripa/Desktop/BallByBall2023(in).csv'
    #     df = pd.read_csv(file_path)
        
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
    with breakdown by bowl type.

    Args:
    df: DataFrame containing cricket match data with NewCommentry, BatsManName, 
        ActualRuns, BallNo, ShotType, XLanding, YLanding, and BowlTypeName columns.

    Returns:
    Tuple of DataFrames:
    - Aggregated analysis of runs scored, shot type, and response metrics by bowl type.
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
        (analysis_df['prev_prev_dot'] == 1))
    
    # Get the results for balls after consecutive dots
    results = analysis_df[analysis_df['after_consecutive_dots']].copy()

    # If no results found, return empty DataFrames
    if len(results) == 0:
        empty_aggregated = pd.DataFrame(columns=[
            'BatsManName', 'BowlTypeName', 'INSTANCES', 'TOTAL_RUNS', 'AVG_RUNS',
            'MOST_COMMON_SHOT', 'STRIKE_RATE', 'SHOT_BREAKDOWN', 'RUNS_BREAKDOWN',
            'TOTAL_CONSECUTIVE_DOT_INSTANCES', 'PERCENTAGE_OF_TOTAL_INSTANCES'
        ])
        empty_raw = pd.DataFrame(columns=['BatsManName', 'XLanding', 'YLanding', 'ActualRuns', 'ShotType'])
        return empty_aggregated, empty_raw, pd.DataFrame()

    # Function to safely get most common shot type
    def get_most_common_shot(x):
        counts = x.value_counts()
        return counts.index[0] if len(counts) > 0 else 'NA'
    
    # Extract unique wicket types only for rows where IsWicket == 1
    wicket_types = results[results['IsWicket'] == 1].groupby(['BatsManName', 'BowlTypeName'])['WicketType'] \
        .apply(lambda x: list(x.unique())).reset_index()
    wicket_types.columns = ['BatsManName', 'BowlTypeName', 'WICKET_TYPES']

    # Aggregations - now grouping by both BatsManName and BowlTypeName
    instances = results.groupby(['BatsManName', 'BowlTypeName'])['after_consecutive_dots'].count()
    total_runs = results.groupby(['BatsManName', 'BowlTypeName'])['ActualRuns'].sum()
    total_wickets = results.groupby(['BatsManName', 'BowlTypeName'])['IsWicket'].sum()
    avg_runs = results.groupby(['BatsManName', 'BowlTypeName'])['ActualRuns'].mean()
    most_common_shots = results.groupby(['BatsManName', 'BowlTypeName'])['ShotType'].agg(get_most_common_shot)

    # Create final aggregated DataFrame
    aggregated_results = pd.DataFrame({
        'INSTANCES': instances,
        'TOTAL_RUNS': total_runs,
        'AVG_RUNS': avg_runs,
        'MOST_COMMON_SHOT': most_common_shots,
        'Total_Wickets': total_wickets
    }).reset_index()

    # Calculate strike rate (runs per ball * 100)
    aggregated_results['STRIKE_RATE'] = (aggregated_results['AVG_RUNS'] * 100).round(2)
    aggregated_results['AVG_RUNS'] = aggregated_results['AVG_RUNS'].round(2)
    
    # Merge with wicket types
    aggregated_results = aggregated_results.merge(wicket_types, on=['BatsManName', 'BowlTypeName'], how='left')

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

    shot_breakdown = results.groupby(['BatsManName', 'BowlTypeName']).apply(get_shot_breakdown).reset_index()
    shot_breakdown.columns = ['BatsManName', 'BowlTypeName', 'SHOT_BREAKDOWN']

    runs_breakdown = results.groupby(['BatsManName', 'BowlTypeName']).apply(get_runs_breakdown).reset_index()
    runs_breakdown.columns = ['BatsManName', 'BowlTypeName', 'RUNS_BREAKDOWN']

    # Convert dictionary to string for display in table
    shot_breakdown['SHOT_BREAKDOWN'] = shot_breakdown['SHOT_BREAKDOWN'].apply(lambda x: json.dumps(x))
    runs_breakdown['RUNS_BREAKDOWN'] = runs_breakdown['RUNS_BREAKDOWN'].apply(lambda x: json.dumps(x))

    # Merge breakdowns with final results
    aggregated_results = aggregated_results.merge(shot_breakdown, on=['BatsManName', 'BowlTypeName'])
    aggregated_results = aggregated_results.merge(runs_breakdown, on=['BatsManName', 'BowlTypeName'])

    # Add total dot ball statistics
    total_consecutive_dots = len(results)
    aggregated_results['TOTAL_CONSECUTIVE_DOT_INSTANCES'] = total_consecutive_dots
    aggregated_results['PERCENTAGE_OF_TOTAL_INSTANCES'] = (
        (aggregated_results['INSTANCES'] / total_consecutive_dots * 100).round(2))

    # Create raw coordinates DataFrame
    results1 = results.dropna(subset=['BatType'])
    results1['IsWicket'] = results1['IsWicket'].astype('int')

    raw_coordinates = results1[['BatsManName', 'BowlTypeName', 'XLanding', 'YLanding', 'ActualRuns', 'BatType', 'ShotType', 'IsWicket']].copy()
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

   # Extract unique BowlType names from the dataset
bowl_types = sorted([str(btype) for btype in df['BowlTypeName'].unique() if btype is not None and btype != 'nan'])

# Dash layout setup
app2.layout = html.Div(className='dashboard-container', children=[
    # Header
    html.Div(className='header', children=[
        html.H1("Cricket Post-Dot Ball Response Analysis", style={'margin': '0', 'textAlign': 'center'}),
        html.P("Analyze how players respond when facing a ball after two consecutive dots", 
               style={'textAlign': 'center', 'marginTop': '10px', 'opacity': '0.8'})
    ]),

 

# Update the control panel card in your layout
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
        
        # Bowl Type Dropdown
        html.Div(style={'flex': '2', 'minWidth': '250px'}, children=[
            html.Label("Select Bowl Type:", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
            dcc.Dropdown(
                id='bowl-type-dropdown',
                options=[{'label': 'All Bowl Types', 'value': 'all'}] + 
                        [{'label': btype, 'value': btype} for btype in bowl_types],
                value='all',
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

@app2.callback(
    [Output('performance-table', 'data'),
     Output('runs-chart', 'figure'),
     Output('strike-rate-chart', 'figure'),
     Output('runs-distribution-chart', 'figure'),
     Output('wagon-wheel-chart', 'figure')],
    [Input('view-type', 'value'),
     Input('player-dropdown', 'value'),
     Input('bowl-type-dropdown', 'value')]
)
def update_table_and_charts(view_type, selected_player, selected_bowl_type):
    # First get all data
    aggregated_df, raw_coordinates, raw_coordinates_with_wickets = analyze_post_dot_ball_response(df)
    
    # Filter by bowl type if not 'all'
    if selected_bowl_type != 'all':
        aggregated_df = aggregated_df[aggregated_df['BowlTypeName'] == selected_bowl_type]
        raw_coordinates = raw_coordinates[raw_coordinates['BowlTypeName'] == selected_bowl_type]
        raw_coordinates_with_wickets = raw_coordinates_with_wickets[raw_coordinates_with_wickets['BowlTypeName'] == selected_bowl_type]
    
    if view_type == 'all':
        # Analyze all data (already filtered by bowl type if applicable)
        wagon_wheel_chart = go.Figure()  # Empty figure for all players view
    else:
        # Filter data for the selected player
        if selected_player:
            aggregated_df = aggregated_df[aggregated_df['BatsManName'] == selected_player]
            raw_coordinates = raw_coordinates[raw_coordinates['BatsManName'] == selected_player]
            raw_coordinates_with_wickets = raw_coordinates_with_wickets[raw_coordinates_with_wickets['BatsManName'] == selected_player]
            
            # Generate wagon wheel for individual player
            if not raw_coordinates.empty:
                raw_df_clean = raw_coordinates.dropna(subset=['XLanding', 'YLanding', 'ActualRuns'])
                wagon_wheel_chart = plot_combined_wagon_wheel(
                    raw_df_clean,
                    x_col='YLanding',
                    y_col='XLanding',
                    player_name='BatsManName',
                    player_batting_types='BatType',
                    title=f'Wagon Wheel - {selected_player}' + 
                         (f' ({selected_bowl_type})' if selected_bowl_type != 'all' else '')
                )
            else:
                wagon_wheel_chart = go.Figure()  # Empty figure if no data
        else:
            # If no player selected, return empty data
            aggregated_df = pd.DataFrame(columns=[
                'BatsManName', 'BowlTypeName', 'INSTANCES', 'TOTAL_RUNS', 'AVG_RUNS', 'MOST_COMMON_SHOT', 
                'STRIKE_RATE', 'SHOT_BREAKDOWN', 'RUNS_BREAKDOWN',
                'TOTAL_CONSECUTIVE_DOT_INSTANCES', 'PERCENTAGE_OF_TOTAL_INSTANCES'
            ])
            wagon_wheel_chart = go.Figure()  # Empty figure

    # Convert DataFrame to dictionary for table
    table_data = aggregated_df.to_dict('records')

    # Create charts (similar to before but now includes bowl type in titles and groupings)
    if len(aggregated_df) > 0:
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
                title='Total Runs After Consecutive Dot Balls' + 
                     (f' ({selected_bowl_type})' if selected_bowl_type != 'all' else ''),
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
                title='Strike Rate After Consecutive Dot Balls' + 
                     (f' ({selected_bowl_type})' if selected_bowl_type != 'all' else ''),
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
            if selected_bowl_type == 'all':
                # Show breakdown by bowl type if "All Bowl Types" selected
                plot_data = aggregated_df
                
                # Total Runs Chart by Bowl Type
                runs_chart = px.bar(
                    plot_data, 
                    x='BowlTypeName', 
                    y='TOTAL_RUNS',
                    title=f"{selected_player}'s Runs by Bowl Type",
                    labels={'BowlTypeName': 'Bowl Type', 'TOTAL_RUNS': 'Total Runs'},
                    color='TOTAL_RUNS',
                    color_continuous_scale=px.colors.sequential.Blues,
                    text='TOTAL_RUNS'
                )
                runs_chart.update_traces(texttemplate='%{text}', textposition='outside')
                runs_chart.update_layout(**plot_layout)
                
                # Strike Rate Chart by Bowl Type
                strike_rate_chart = px.bar(
                    plot_data,
                    x='BowlTypeName', 
                    y='STRIKE_RATE',
                    title=f"{selected_player}'s Strike Rate by Bowl Type",
                    labels={'BowlTypeName': 'Bowl Type', 'STRIKE_RATE': 'Strike Rate'},
                    color='STRIKE_RATE',
                    color_continuous_scale=px.colors.sequential.Greens,
                    text='STRIKE_RATE'
                )
                strike_rate_chart.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                strike_rate_chart.update_layout(**plot_layout)
                
                # Pie chart for runs distribution across all bowl types
                runs_distribution = raw_coordinates.groupby('ActualRuns').size().reset_index(name='Count')
                runs_distribution['Label'] = runs_distribution['ActualRuns'].map({
                    0: 'Dot Ball',
                    1: 'Single', 
                    2: 'Double',
                    3: 'Triple',
                    4: 'Boundary',
                    6: 'Six'
                })
                
                runs_distribution_chart = px.pie(
                    runs_distribution,
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
                # For specific bowl type selected, show individual player's performance
                player_row = aggregated_df.iloc[0] if len(aggregated_df) > 0 else None
                
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
                        title=f"{selected_player}'s Shot Selection ({selected_bowl_type})",
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
                        title=f'Strike Rate Comparison ({selected_bowl_type})',
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
                        title=f"{selected_player}'s Runs Distribution ({selected_bowl_type})",
                        color_discrete_sequence=px.colors.sequential.RdBu,
                        hole=0.4,
                    )
                    runs_distribution_chart.update_layout(
                        title={'text': f"{selected_player}'s Runs Distribution ({selected_bowl_type})", 'x': 0.5, 'xanchor': 'center'},
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

import os
import pandas as pd

# Get the current directory of the running script
#current_dir = os.path.expanduser("~/Desktop")

# Build the relative path to the CSV file
file_path = os.path.join(current_dir, "BallByBall2024(in).csv")

# Load the CSV file
df = pd.read_csv(file_path)


# file_path = 'C:/Users/kripa/Desktop/BallByBall2023(in).csv'
# df = pd.read_csv(file_path)

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



# Function to analyze post-boundary bowler response (enhanced for multiple bowlers)
def analyze_post_boundary_response(df, bowler_names=None, grouping_columns=None):
    """
    Analyze how bowlers respond when bowling a ball after conceding a boundary (four or six),
    with flexible grouping options and support for multiple bowlers.
    
    Args:
    df: DataFrame containing cricket match data with relevant columns.
    bowler_names: List of bowler names to filter by. If None, analyzes all bowlers.
    grouping_columns: List of column names to group by. 
                      Defaults to ['BowlerName', 'BowlTypeName', 'OverName'] if None.
    
    Returns:
    Tuple of (aggregated results DataFrame, filtered raw results DataFrame)
    """
    # Create a copy to avoid modifying original
    analysis_df = df.copy()
    
    # Filter by bowler names if provided
    if bowler_names is not None:
        if isinstance(bowler_names, str):
            bowler_names = [bowler_names]
        analysis_df = analysis_df[analysis_df['BowlerName'].isin(bowler_names)]
    
    # Ensure numeric types
    analysis_df['ActualRuns'] = pd.to_numeric(analysis_df['ActualRuns'], errors='coerce')
    
    # Default grouping if not provided
    if grouping_columns is None:
        grouping_columns = ['BowlerName', 'BowlTypeName', 'OverName']
    
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
        (final_results['INSTANCES'] / total_boundary_instances * 100).round(2))
    
    return final_results.sort_values('INSTANCES', ascending=False), results

# If Xpitch and Ypitch don't exist, create them with placeholder values for visualization
if 'Xpitch_meters' not in df.columns:
    # Creating random X values between -1.5 and 1.5 (typical pitch width in meters)
    import numpy as np
    df['Xpitch_meters'] = np.random.uniform(-1.5, 1.5, len(df))

if 'Ypitch_meters' not in df.columns:
    # Creating random Y values between 0 and 20 (typical pitch length in meters)
    df['Ypitch_meters'] = np.random.uniform(0, 20, len(df))
    
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
        
        # Color by bowler if multiple bowlers are present
        if 'BowlerName' in df.columns and len(df['BowlerName'].unique()) > 1:
            # Create a color mapping for bowlers
            bowlers = df['BowlerName'].unique()
            color_map = {bowler: idx for idx, bowler in enumerate(bowlers)}
            color_values = df['BowlerName'].map(color_map)
            
            fig.add_trace(go.Scatter3d(
                x=df['Xpitch_meters'],
                y=df['Ypitch_meters'],
                z=z_values,
                mode='markers',
                marker=dict(
                    size=10,
                    color=color_values,
                    colorscale='Viridis',
                    colorbar=dict(title='Bowler'),
                opacity=0.8,
                symbol='circle',
            ),
            hovertemplate='<b>Bowler:</b> %{text}<br>' +
                         '<b>Runs:</b> %{marker.color}<br>' +
                         '<b>Line:</b> %{x:.2f}<br>' +
                         '<b>Length:</b> %{y:.2f}<br>' +
                         '<b>Height:</b> %{z:.2f}<extra></extra>',
            text=df['BowlerName'],
            name='Deliveries'
            ))
        else:
            # Default coloring by runs
            color_values = df['ActualRuns'] if 'ActualRuns' in df.columns else np.ones(len(df))
            
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
                        {'label': html.Span([html.I(className="fas fa-user", style={'marginRight': '5px'}), 'Compare Bowlers (up to 5)']), 'value': 'individual'},
                    ],
                    value='all',
                    style={'marginTop': '8px'},
                    labelStyle={'marginRight': '15px', 'display': 'inline-block'}
                ),
            ]),
            
            # Bowler Dropdown
            html.Div(id='bowler-selection-container', style={'flex': '2', 'minWidth': '250px'}, children=[
                html.Label("Select Bowler(s):", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='bowler-dropdown',
                    options=[{'label': bowler, 'value': bowler} for bowler in bowler_names],
                    value=[bowler_names[0]] if bowler_names else None,
                    multi=True,
                    style={'width': '100%'},
                    placeholder="Select up to 5 bowlers...",
                )
            ]),
            
            # OverName Dropdown Container
            html.Div(id='over-dropdown-container', style={'flex': '1', 'minWidth': '250px'}, children=[
                html.Label("Select Over:", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='over-dropdown',
                    options=[{'label': over, 'value': over} for over in df['OverName'].unique()],
                    value=None,
                    style={'width': '100%'}
                )
            ]),

            # BowlTypeName Dropdown Container
            html.Div(id='bowl-type-dropdown-container', style={'flex': '1', 'minWidth': '250px'}, children=[
                html.Label("Select Bowl Type:", style={'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='bowl-type-dropdown',
                    options=[{'label': bowl_type, 'value': bowl_type} for bowl_type in df['BowlTypeName'].unique()],
                    value=None,
                    style={'width': '100%'}
                )
            ]),
        ]),
    ]),
                
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
                value='all',
                labelStyle={'marginRight': '15px', 'display': 'inline-block'}
            )
        ]),
        
        # Pitch Map Visualization
        dcc.Graph(id='pitch-map', style={'height': '600px'})
    ]),
])

# Callback to limit bowler selection to 5
@app.callback(
    Output('bowler-dropdown', 'value'),
    [Input('bowler-dropdown', 'value')]
)
def limit_bowler_selection(selected_bowlers):
    if selected_bowlers and len(selected_bowlers) > 5:
        return selected_bowlers[:5]
    return selected_bowlers

# Callback to show/hide bowler dropdown based on view type
@app.callback(
    Output('bowler-selection-container', 'style'),
    [Input('view-type', 'value')]
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
    [Input('view-type', 'value')]
)
def toggle_pitch_filter(view_type):
    if view_type == 'individual':
        return {'marginBottom': '15px', 'display': 'block'}
    else:
        return {'display': 'none'}

# Callback to show/hide individual bowler charts
@app.callback(
    Output('individual-charts-container', 'style'),
    [Input('view-type', 'value')]
)
def toggle_individual_charts(view_type):
    if view_type == 'individual':
        return {'display': 'block', 'marginTop': '20px'}
    else:
        return {'display': 'none'}

# Main callback to update all dashboard components
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
def update_dashboard(view_type, selected_bowlers, over_name, bowl_type, outcome_filter):
    # Start with the full dataset
    df_filtered = df.copy()

    # Filter data based on OverName
    if over_name:
        df_filtered = df_filtered[df_filtered['OverName'] == over_name]

    # Filter data based on BowlTypeName
    if bowl_type:
        df_filtered = df_filtered[df_filtered['BowlTypeName'] == bowl_type]

    # Analyze data based on view type
    if view_type == 'all':
        # Analyze all bowlers with the applied filters
        analysis_result, detailed_results = analyze_post_boundary_response(df_filtered)
        analysis_result1, detailed_results1 = analyze_post_boundary_response(df_filtered, grouping_columns=['BowlerName', 'BowlTypeName'])
        analysis_result2, detailed_results2 = analyze_post_boundary_response(df_filtered, grouping_columns=['BowlerName', 'OverName'])
    else:
        # Filter for the selected bowlers with the applied filters
        if selected_bowlers:
            analysis_result, detailed_results = analyze_post_boundary_response(
                df_filtered, 
                bowler_names=selected_bowlers,
                grouping_columns=['BowlerName', 'BowlTypeName', 'OverName']
            )
            analysis_result1, detailed_results1 = analyze_post_boundary_response(
                df_filtered,
                bowler_names=selected_bowlers,
                grouping_columns=['BowlerName', 'BowlTypeName']
            )
            analysis_result2, detailed_results2 = analyze_post_boundary_response(
                df_filtered,
                bowler_names=selected_bowlers,
                grouping_columns=['BowlerName', 'OverName']
            )
        else:
            # If no bowler is selected, return empty DataFrames
            return [], px.bar(title="Please select at least one bowler"), px.bar(title="Please select at least one bowler"), px.pie(title="Please select at least one bowler"), create_3d_pitch_map(pd.DataFrame(), title="No Data Available")

    # Convert DataFrame to dictionary for table
    table_data = analysis_result.to_dict('records')

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
            # For individual view with multiple bowlers
            if len(selected_bowlers) > 0:
                # Economy Rate Chart (multiple bars for selected bowlers)
                economy_chart = px.bar(
                    analysis_result1, 
                    x='BowlerName', 
                    y='ECONOMY_RATE',
                    color='BowlTypeName',
                    title='Economy After Conceding a Boundary',
                    labels={'BowlerName': 'Bowler', 'ECONOMY_RATE': 'Economy Rate'},
                    barmode='group',
                    text='INSTANCES'
                )
                economy_chart.update_traces(texttemplate='Instances: %{text}', textposition='outside')
                economy_chart.update_layout(**plot_layout)
                
                # Dot Ball Percentage Chart (multiple bars for selected bowlers)
                dot_ball_chart = px.bar(
                    analysis_result1,
                    x='BowlerName', 
                    y='DOT_BALL_PERCENTAGE',
                    color='BowlTypeName',
                    title='Dot Ball % After Conceding a Boundary',
                    labels={'BowlerName': 'Bowler', 'DOT_BALL_PERCENTAGE': 'Dot Ball %'},
                    barmode='group',
                    text='INSTANCES'
                )
                dot_ball_chart.update_traces(texttemplate='Instances: %{text}', textposition='outside')
                dot_ball_chart.update_layout(**plot_layout)
                
                # Bowl Type Chart (multiple bars for selected bowlers)
                bowl_type_chart = px.bar(
                    analysis_result1,
                    x='BowlerName', 
                    y='TOTAL_RUNS_CONCEDED',
                    color='BowlTypeName',
                    title='Runs Conceded After a Boundary',
                    labels={'BowlerName': 'Bowler', 'TOTAL_RUNS_CONCEDED': 'Total Runs Conceded'},
                    barmode='group',
                    text='INSTANCES'
                )
                bowl_type_chart.update_traces(texttemplate='Instances: %{text}', textposition='outside')
                bowl_type_chart.update_layout(**plot_layout)
                
                # Runs Distribution Chart (pie chart for selected bowlers)
                if not analysis_result2.empty:
                    runs_distribution_chart = px.pie(
                        analysis_result2,
                        names='OverName', 
                        values='TOTAL_RUNS_CONCEDED',
                        title='Runs Distribution After Boundary',
                        color_discrete_sequence=px.colors.sequential.RdBu,
                        facet_col='BowlerName',  # Separate pie charts for each bowler
                        facet_col_wrap=min(3, len(selected_bowlers))  # Max 3 columns
                    )
                    runs_distribution_chart.update_layout(**plot_layout)
                
                # Filter based on selected outcome
                if outcome_filter == 'dot':
                    detailed_results = detailed_results[detailed_results['ActualRuns'] == 0]
                elif outcome_filter == 'runs':
                    detailed_results = detailed_results[detailed_results['ActualRuns'] > 0]
                elif outcome_filter == 'wicket':
                    detailed_results = detailed_results[detailed_results['IsWicket'] == 1]
                elif outcome_filter == 'boundary':
                    detailed_results = detailed_results[(detailed_results['IsFour'] == 1) | (detailed_results['IsSix'] == 1)]
                
                # Pitch Map for Multiple Bowlers
                if not detailed_results.empty and all(col in detailed_results.columns for col in ['Xpitch_meters', 'Ypitch_meters']):
                    pitch_map = create_3d_pitch_map(
                        detailed_results, 
                        title=f"Pitch Map - {', '.join(selected_bowlers)}"
                    )
                else:
                    pitch_map = create_3d_pitch_map(pd.DataFrame(), title="No Data Available")

    return table_data, economy_chart, dot_ball_chart, bowl_type_chart, runs_distribution_chart, pitch_map



# Replace 'YourUsername' with your actual username and 'data.csv' with your actual file name
#file_path = 'C:/Users/kripa/Desktop/IPL_23_24.csv'

## Get the current directory of the running script
#current_dir = os.path.expanduser("~/Desktop")
    
# Build the relative path to the CSV files
file_path1 = os.path.join(current_dir, "IPL_23_HE_part1.csv")
file_path2 = os.path.join(current_dir, "IPL_23_HE_part2.csv")
file_path3 = os.path.join(current_dir, "IPL_23_HE_part3.csv")
file_path4 = os.path.join(current_dir, "IPL_23_HE_part4.csv")

# Load and combine data
df_part1 = pd.read_csv(file_path1)
df_part2 = pd.read_csv(file_path2)
df_part3 = pd.read_csv(file_path3)
df_part4 = pd.read_csv(file_path4)

combined_df = pd.concat([df_part1, df_part2, df_part3, df_part4], ignore_index=True)

def process_cricket_data(df):
    """
    Process cricket data and create processed dataframe with required transformations.
    
    Args:
        df (pd.DataFrame): Raw cricket data
        
    Returns:
        pd.DataFrame: Processed data ready for visualization
    """
    # Convert MPH to KPH (handle null values)
    df['Speed'] = df['match.delivery.trajectory.releaseSpeed'].astype(float) * 1.60934

    # Extract venue (stadium) from text
    def extract_venue(text):
        if isinstance(text, str):
            match = re.search(r'_([^_]*)-(?!.*-)', text)
            if match:
                return match.group(1)
        return 'Unknown'
    
    df['VENUE'] = df['match.name'].apply(extract_venue)

    # Create over segment
    def get_over_segment(over):
        try:
            over = float(over)
            if over <= 5:
                return 'POWERPLAY'
            elif 6 <= over <= 10:
                return 'MP1'
            elif 11 <= over <= 15:
                return 'MP2'
            else:
                return 'DEATH'
        except:
            return 'Unknown'
    
    df['OVER_SEGMENT'] = df['match.delivery.deliveryNumber.over'].apply(get_over_segment)
    
    # Remove negative scores and invalid data
    processed_df = df[
        (df['match.delivery.scoringInformation.score'] >= 0) & 
        (df['match.delivery.deliveryNumber.ball'] > 0)
    ].copy()
    
    # Calculate strike rate (handle division by zero)
    processed_df['STRIKE_RATE'] = (
        processed_df['match.delivery.scoringInformation.score'] / 
        processed_df['match.delivery.deliveryNumber.ball'].replace(0, np.nan) * 100
    ).round(2)
    
    # Rename columns for clarity
    processed_df = processed_df.rename(columns={
        'match.battingTeam.batsman.name': 'BATSMAN',
        'match.delivery.deliveryNumber.ball': 'BALLS_FACED',
        'match.delivery.scoringInformation.score': 'RUNS_SCORED',
        'match.delivery.deliveryType': 'DELIVERY_TYPE',
        'match.delivery.deliveryNumber.innings': 'INNINGS_NUMBER'
    })
    
    # Add cumulative totals by batsman
    processed_df['TOTAL_BALLS'] = processed_df.groupby('BATSMAN')['BALLS_FACED'].cumcount()
    processed_df['TOTAL_RUNS'] = processed_df.groupby('BATSMAN')['RUNS_SCORED'].cumsum()
    
    # Define speed bins and labels
    bins = [0, 80, 85, 90, 120, 130, 140, 150, np.inf]
    labels = ['<80', '81-85', '86-90', '91-120', '121-130', '131-140', '141-150', '>150']

    # Assign speed categories (handle null values)
    processed_df['SPEED_CATEGORY'] = pd.cut(
        processed_df['Speed'].fillna(0), 
        bins=bins, 
        labels=labels, 
        right=False
    )
    
    return processed_df

# Process data
processed_data = process_cricket_data(combined_df)

def create_speed_dashboard(processed_data):
    """Create and return the speed dashboard app with all its components."""
    
    # Initialize Dash app with Bootstrap theme
    app3 = dash.Dash(
        __name__,
        server=server,
        url_base_pathname="/speed-dashboard/",
        external_stylesheets=[
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css', 
            dbc.themes.BOOTSTRAP
        ]
    )

    # Pre-process the over segment column to ensure consistent format
    processed_data['match.delivery.deliveryNumber.over'] = processed_data['match.delivery.deliveryNumber.over'].apply(
        lambda x: str(int(float(x))) if pd.notnull(x) else 'Unknown'
    )
    processed_data['DELIVERY_TYPE'] = processed_data['DELIVERY_TYPE'].fillna('Unknown')

    # Filters for dropdowns
    filters = {
        'over_segments': sorted([str(int(float(x))) for x in processed_data['match.delivery.deliveryNumber.over'].unique() if pd.notnull(x) and x != 'Unknown']),
        'years': sorted(processed_data['YEAR'].dropna().unique()),
        'innings': sorted(processed_data['INNINGS_NUMBER'].dropna().unique()),
        'delivery_types': sorted(processed_data['DELIVERY_TYPE'].dropna().unique()),
        'speed_categories': sorted(processed_data['SPEED_CATEGORY'].dropna().unique()),
        'batsmen': sorted(processed_data['BATSMAN'].dropna().unique())
    }

    # Dashboard layout
    app3.layout = dbc.Container([
        dbc.Row([dbc.Col(html.H1("Speed Analytics Dashboard", className="text-center text-primary my-4"))]),
        
        # Filters Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Filters", className="bg-primary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Batsman"),
                                dcc.Dropdown(
                                    id='batsman-dropdown',
                                    options=[{'label': b, 'value': b} for b in filters['batsmen']],
                                    multi=True,
                                    placeholder="Select Batsmen"
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Year"),
                                dcc.Dropdown(
                                    id='year-dropdown',
                                    options=[{'label': str(y), 'value': y} for y in filters['years']],
                                    multi=True,
                                    placeholder="Select Years"
                                )
                            ], width=6),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Over Segment"),
                                dcc.Dropdown(
                                    id='over-segment-dropdown',
                                    options=[{'label': s, 'value': s} for s in filters['over_segments']],
                                    multi=True,
                                    placeholder="Select Segments"
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Delivery Type"),
                                dcc.Dropdown(
                                    id='delivery-type-dropdown',
                                    options=[{'label': d, 'value': d} for d in filters['delivery_types']],
                                    multi=True,
                                    placeholder="Select Delivery Types"
                                )
                            ], width=6),
                        ]),
                        dbc.Row([  
                            dbc.Col([  
                                html.Label("Speed Category"),  
                                dcc.Dropdown(  
                                    id='speed-category-dropdown',  
                                    options=[{'label': sc, 'value': sc} for sc in filters['speed_categories']],  
                                    multi=True,  
                                    placeholder="Select Speed Categories"  
                                )  
                            ], width=6),  
                        ]),
                        dbc.Button("Apply Filters", id="apply-filters", color="primary", className="mt-2")
                    ])
                ])
            ])
        ]),

        # Graphs
        dbc.Row([
            dbc.Col(dcc.Graph(id='performance-chart'), width=8),
            dbc.Col(dcc.Graph(id='speed-distribution'), width=4)
        ]),

        dbc.Row([dbc.Col(dash_table.DataTable(id='stats-table', style_table={'overflowX': 'auto'}))]),

        # Trends tab
        dbc.Row([dbc.Col(dcc.Graph(id='trend-chart'))]),
    ], fluid=True)

    @app3.callback(
        [Output('performance-chart', 'figure'),
         Output('speed-distribution', 'figure'),
         Output('trend-chart', 'figure'),
         Output('stats-table', 'data'),
         Output('stats-table', 'columns')],
        [Input('apply-filters', 'n_clicks')],
        [State('batsman-dropdown', 'value'),
         State('year-dropdown', 'value'),
         State('over-segment-dropdown', 'value'),
         State('delivery-type-dropdown', 'value'),
         State('speed-category-dropdown', 'value')]
    )
    def update_dashboard(n_clicks, batsmen, years, segments, delivery_types, speed_categories):
        # Filter data
        filtered_df = processed_data.copy()

        # Step 1: Ensure correct data types and handle NaN values
        filtered_df['BATSMAN'] = filtered_df['BATSMAN'].astype(str)
        filtered_df['YEAR'] = filtered_df['YEAR'].astype(int)
        filtered_df['match.delivery.deliveryNumber.over'] = filtered_df['match.delivery.deliveryNumber.over'].astype(str)
        filtered_df['DELIVERY_TYPE'] = filtered_df['DELIVERY_TYPE'].astype(str)
        filtered_df['SPEED_CATEGORY'] = filtered_df['SPEED_CATEGORY'].astype(str)

        # Apply filters 
        if batsmen:
            filtered_df = filtered_df[filtered_df['BATSMAN'].isin(batsmen)]
        if years:
            filtered_df = filtered_df[filtered_df['YEAR'].isin(years)]
        if segments:
            # Convert segments to strings and handle the filtering properly
            segments = [str(seg) for seg in segments]
            filtered_df = filtered_df[filtered_df['match.delivery.deliveryNumber.over'].isin(segments)]
        if delivery_types:
            filtered_df = filtered_df[filtered_df['DELIVERY_TYPE'].isin(delivery_types)]
        if speed_categories:
            filtered_df = filtered_df[filtered_df['SPEED_CATEGORY'].isin(speed_categories)]

        # Convert columns to numeric where necessary
        filtered_df['RUNS_SCORED'] = pd.to_numeric(filtered_df['RUNS_SCORED'], errors='coerce')
        filtered_df['BALLS_FACED'] = pd.to_numeric(filtered_df['BALLS_FACED'], errors='coerce')

        # Group and calculate performance
        grouped_df = filtered_df.groupby(
            ['BATSMAN', 'YEAR', 'match.delivery.deliveryNumber.over', 'DELIVERY_TYPE', 'SPEED_CATEGORY'],
            as_index=False
        ).agg({'RUNS_SCORED': 'sum', 'BALLS_FACED': 'count'})

        # Calculate strike rate
        grouped_df['STRIKE_RATE'] = (grouped_df['RUNS_SCORED'] / grouped_df['BALLS_FACED'].replace(0, np.nan) * 100).round(2)
        grouped_df = grouped_df.dropna(subset=['STRIKE_RATE'])

        # Calculate average strike rate by speed category
        avg_strike_rate_df = grouped_df.groupby('SPEED_CATEGORY', as_index=False).agg({'STRIKE_RATE': 'mean'})

        # Create plots
        perf_chart = px.bar(avg_strike_rate_df, x='SPEED_CATEGORY', y='STRIKE_RATE', 
                           title="Speed Category vs Average Strike Rate", 
                           color='SPEED_CATEGORY', template='plotly_white')
        
        speed_dist_chart = px.histogram(grouped_df, x='SPEED_CATEGORY', 
                                      title="Speed Category Distribution", 
                                      color='SPEED_CATEGORY', template='plotly_white')
        
        trend_chart = px.line(grouped_df, x='YEAR', y='STRIKE_RATE', 
                            color='BATSMAN', title="Performance Trends")

        # Convert data to table format
        table_data = grouped_df.to_dict('records')
        table_columns = [{'name': col, 'id': col} for col in grouped_df.columns]

        return perf_chart, speed_dist_chart, trend_chart, table_data, table_columns

    return app3

# Load processed data (replace with your data loading method)
processed_data1 = processed_data.copy()  # Placeholder, replace with actual data loading

# Create Speed Dashboard
app3 = create_speed_dashboard(processed_data1)

# Define the home route (must come AFTER Dash apps are initialized)
@server.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cricket Analytics</title>
        <style>
            body {
                font-family: "Segoe UI", Arial, sans-serif;
                background-color: #f5f5f5;
                color: #333;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                text-align: center;
                background-color: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                max-width: 600px;
            }
            h1 {
                color: #1976D2;
                margin-top: 0;
            }
            p {
                margin: 20px 0;
                line-height: 1.6;
            }
            .btn-container {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            .btn {
                display: inline-block;
                background-color: #1976D2;
                color: white;
                text-decoration: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-weight: bold;
                transition: background-color 0.3s;
            }
            .btn:hover {
                background-color: #1565C0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Cricket Analytics Dashboard</h1>
            <p>Explore detailed analytics for bowlers, batsmen, and bowling speeds.</p>
            <div class="btn-container">
                <a href="/bowler-dashboard/" class="btn">Bowler Dashboard</a>
                <a href="/batsman-dashboard/" class="btn">Batsman Dashboard</a>
                <a href="/speed-dashboard/" class="btn">Speed Dashboard</a>
            </div>
        </div>
    </body>
    </html>
    """


# if __name__ == "__main__":
#     server.run(debug=True, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use the port assigned by Render
    app.run(host="0.0.0.0", port=port)
