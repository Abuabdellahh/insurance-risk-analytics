"""
Visualization Module

Provides tools for creating interactive and static visualizations for insurance risk analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class InsuranceVisualizer:
    """
    Insurance Risk Visualization Tools
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Visualizer
        
        Args:
            data: Input DataFrame for visualization
        """
        self.data = data
        self.palette = sns.color_palette("husl", 8)
        
    def plot_loss_ratio_by_province(self, province_col: str, loss_col: str) -> go.Figure:
        """
        Create interactive bar chart showing loss ratio by province
        
        Args:
            province_col: Province column name
            loss_col: Loss ratio column name
            
        Returns:
            Plotly Figure object
        """
        fig = px.bar(
            self.data.groupby(province_col)[loss_col].mean().reset_index(),
            x=province_col,
            y=loss_col,
            title='Loss Ratio by Province',
            color=province_col,
            labels={province_col: 'Province', loss_col: 'Loss Ratio'},
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title='Province',
            yaxis_title='Loss Ratio',
            showlegend=False
        )
        
        return fig
    
    def plot_claim_severity_distribution(self, severity_col: str) -> go.Figure:
        """
        Create histogram of claim severity distribution
        
        Args:
            severity_col: Claim severity column name
            
        Returns:
            Plotly Figure object
        """
        fig = px.histogram(
            self.data,
            x=severity_col,
            title='Distribution of Claim Severity',
            marginal='box',
            nbins=30,
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title='Claim Severity',
            yaxis_title='Frequency'
        )
        
        return fig
    
    def plot_risk_heatmap(self, x_col: str, y_col: str, value_col: str) -> go.Figure:
        """
        Create heatmap showing risk distribution
        
        Args:
            x_col: X-axis column name
            y_col: Y-axis column name
            value_col: Value column name
            
        Returns:
            Plotly Figure object
        """
        heatmap_data = self.data.groupby([x_col, y_col])[value_col].mean().unstack()
        
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn_r'
            )
        )
        
        fig.update_layout(
            title='Risk Distribution Heatmap',
            xaxis_title=x_col,
            yaxis_title=y_col,
            template='plotly_white'
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importances: Dict, top_n: int = 10) -> go.Figure:
        """
        Create bar chart showing feature importance
        
        Args:
            feature_importances: Dictionary of feature names and their importances
            top_n: Number of top features to show
            
        Returns:
            Plotly Figure object
        """
        sorted_importances = sorted(
            feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        features, importances = zip(*sorted_importances)
        
        fig = go.Figure(
            go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h'
            )
        )
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Features',
            template='plotly_white'
        )
        
        return fig
    
    def plot_time_series(self, date_col: str, value_col: str) -> go.Figure:
        """
        Create time series plot
        
        Args:
            date_col: Date column name
            value_col: Value column name
            
        Returns:
            Plotly Figure object
        """
        fig = px.line(
            self.data.sort_values(date_col),
            x=date_col,
            y=value_col,
            title=f'Time Series of {value_col}',
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title=value_col
        )
        
        return fig
