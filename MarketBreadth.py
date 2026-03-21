#!/usr/bin/env python
# coding: utf-8

# In[1]:


#market breadth calculator


# In[2]:


import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Configuration
LOOKBACK_YEARS = 27
MOMENTUM_WINDOWS = {
    '1M': 21,   # ~1 month of trading days
    '3M': 63,   # ~3 months
    '6M': 126,  # ~6 months
    '12M': 252  # ~12 months
}
DATA_FILE = 'sp500_breadth_history.csv'
WEIGHTS_FILE = 'sp500_weights_history.csv'

def scrape_sp500_components():
    """Scrape S&P 500 components and weights from slickcharts"""
    print("Scraping S&P 500 components from slickcharts.com...")
    url = "https://www.slickcharts.com/sp500"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    table = soup.find('table')
    rows = table.find_all('tr')[1:]  # Skip header
    
    # Symbol corrections for Yahoo Finance format
    symbol_corrections = {
        'BRK.B': 'BRK-B',
        'BF.B': 'BF-B'
    }
    
    components = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 3:
            symbol = cols[2].text.strip()
            weight_text = cols[3].text.strip().replace('%', '')
            
            # Apply symbol corrections
            if symbol in symbol_corrections:
                corrected = symbol_corrections[symbol]
                print(f"  Correcting symbol: {symbol} -> {corrected}")
                symbol = corrected
            
            try:
                weight = float(weight_text)
                components.append({'symbol': symbol, 'weight': weight})
            except:
                continue
    
    df = pd.DataFrame(components)
    print(f"Found {len(df)} components")
    return df

def download_price_data(symbols, start_date, end_date):
    """Download historical price data for all symbols"""
    print(f"\nDownloading price data from {start_date} to {end_date}...")
    
    # Ticker changes to handle for data continuity
    # Format: (new_ticker, old_ticker, change_date)
    ticker_changes = [
        ('MRSH', 'MMC', '2026-01-14')  # Marsh McLennan ticker change
    ]
    
    all_data = {}
    failed = []
    
    for i, symbol in enumerate(symbols, 1):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(symbols)}")
        
        try:
            # Check if this symbol had a ticker change
            old_symbol = None
            change_date = None
            for new_tick, old_tick, chg_date in ticker_changes:
                if symbol == new_tick:
                    old_symbol = old_tick
                    change_date = pd.to_datetime(chg_date)
                    print(f"  Handling ticker change: {old_symbol} -> {symbol} on {chg_date}")
                    break
            
            if old_symbol:
                # Download historical data using old ticker
                ticker_old = yf.Ticker(old_symbol)
                hist_old = ticker_old.history(start=start_date, end=change_date)
                
                # Download recent data using new ticker
                ticker_new = yf.Ticker(symbol)
                hist_new = ticker_new.history(start=change_date, end=end_date)
                
                # Combine the data
                if not hist_old.empty or not hist_new.empty:
                    hist = pd.concat([hist_old, hist_new])
                    hist = hist[~hist.index.duplicated(keep='last')]  # Remove duplicates
                    hist = hist.sort_index()
                    
                    if len(hist) > 252:
                        all_data[symbol] = hist['Close']
                    else:
                        failed.append(symbol)
                else:
                    failed.append(symbol)
            else:
                # Normal download for symbols without ticker changes
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty and len(hist) > 252:
                    all_data[symbol] = hist['Close']
                else:
                    failed.append(symbol)
                    
        except Exception as e:
            failed.append(symbol)
            continue
    
    print(f"\nSuccessfully downloaded: {len(all_data)} stocks")
    print(f"Failed: {len(failed)} stocks")
    if failed:
        print(f"Failed symbols: {failed[:10]}...")
    
    # Combine into single DataFrame
    prices_df = pd.DataFrame(all_data)
    return prices_df

def calculate_momentum(prices_df, window):
    """Calculate rate of change (momentum) over specified window"""
    # ROC = (Price_today - Price_window_ago) / Price_window_ago
    momentum = (prices_df - prices_df.shift(window)) / prices_df.shift(window)
    return momentum

def calculate_equal_weighted_breadth(prices_df):
    """Calculate equal-weighted market breadth for all windows"""
    breadth_data = {}
    
    for window_name, window_days in MOMENTUM_WINDOWS.items():
        print(f"Calculating {window_name} momentum...")
        momentum = calculate_momentum(prices_df, window_days)
        
        # Count % of stocks with positive momentum each day
        positive_count = (momentum > 0).sum(axis=1)
        total_count = momentum.notna().sum(axis=1)
        breadth_pct = (positive_count / total_count * 100).dropna()
        
        breadth_data[window_name] = breadth_pct
    
    return pd.DataFrame(breadth_data)

def calculate_cap_weighted_breadth(prices_df, weights_df):
    """Calculate market-cap weighted breadth for most recent day"""
    latest_date = prices_df.index[-1] #UPDATE THIS TO -2 IF GETTING WEIRD OUTPUTS
    results = []
    
    # Create weight mapping
    weight_map = dict(zip(weights_df['symbol'], weights_df['weight']))
    
    for window_name, window_days in MOMENTUM_WINDOWS.items():
        momentum = calculate_momentum(prices_df, window_days)
        latest_momentum = momentum.loc[latest_date]
        
        # Calculate weighted breadth
        total_weight = 0
        positive_weight = 0
        
        for symbol in latest_momentum.index:
            if pd.notna(latest_momentum[symbol]) and symbol in weight_map:
                weight = weight_map[symbol]
                total_weight += weight
                if latest_momentum[symbol] > 0:
                    positive_weight += weight
        
        breadth_pct = (positive_weight / total_weight * 100) if total_weight > 0 else 0
        
        results.append({
            'Window': window_name,
            'Date': latest_date.strftime('%Y-%m-%d'),
            'Breadth_Pct': round(breadth_pct, 2),
            'Positive_Weight': round(positive_weight, 2),
            'Total_Weight': round(total_weight, 2)
        })
    
    return pd.DataFrame(results)

def save_cap_weighted_data(new_data):
    """Append new cap-weighted data to historical file"""
    if os.path.exists(DATA_FILE):
        existing = pd.read_csv(DATA_FILE)
        # Remove any existing data for the same date
        existing = existing[existing['Date'] != new_data['Date'].iloc[0]]
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data
    
    combined.to_csv(DATA_FILE, index=False)
    print(f"\nSaved cap-weighted data to {DATA_FILE}")
    return combined

def save_weights_history(weights_df, date_str):
    """Save S&P 500 constituent weights over time"""
    # Create a DataFrame with symbol as index and date as column
    weights_for_date = weights_df.set_index('symbol')['weight']
    weights_for_date.name = date_str
    
    if os.path.exists(WEIGHTS_FILE):
        # Load existing weights history
        existing = pd.read_csv(WEIGHTS_FILE, index_col=0)
        
        # Check if this date already exists
        if date_str in existing.columns:
            print(f"  Updating weights for {date_str} in {WEIGHTS_FILE}")
            existing[date_str] = weights_for_date
            combined = existing
        else:
            print(f"  Adding new date {date_str} to {WEIGHTS_FILE}")
            # Add new column
            combined = existing.copy()
            combined[date_str] = weights_for_date
    else:
        # Create new file
        print(f"  Creating new weights history file: {WEIGHTS_FILE}")
        combined = pd.DataFrame(weights_for_date)
    
    # Sort columns by date
    combined = combined[sorted(combined.columns)]
    
    # Save with symbols as rows, dates as columns
    combined.to_csv(WEIGHTS_FILE)
    print(f"Saved S&P 500 weights history to {WEIGHTS_FILE}")
    print(f"  Total symbols tracked: {len(combined)}")
    print(f"  Total dates recorded: {len(combined.columns)}")
    
    return combined

def plot_equal_weighted_breadth(breadth_df):
    """Create 4 separate interactive plots for equal-weighted breadth"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[f'{window} Momentum Breadth' for window in MOMENTUM_WINDOWS.keys()],
        vertical_spacing=0.08
    )
    
    windows = list(MOMENTUM_WINDOWS.keys())
    
    colors = ['steelblue', 'darkgreen', 'darkorange', 'crimson']
    
    for i, (window, color) in enumerate(zip(windows, colors), 1):
        # Main breadth line
        fig.add_trace(
            go.Scatter(
                x=breadth_df.index,
                y=breadth_df[window],
                name=window,
                line=dict(color=color, width=2),
                showlegend=False,
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                              '<b>Breadth</b>: %{y:.1f}%<br>' +
                              '<extra></extra>'
            ),
            row=i, col=1
        )
        
        # 50% reference line
        fig.add_hline(
            y=50, line_dash="dash", line_color="red", line_width=1,
            opacity=0.7, row=i, col=1
        )
        
        # Shaded regions for extreme readings
        fig.add_hrect(
            y0=0, y1=10, fillcolor="red", opacity=0.1,
            line_width=0, row=i, col=1
        )
        fig.add_hrect(
            y0=90, y1=100, fillcolor="green", opacity=0.1,
            line_width=0, row=i, col=1
        )
    
    # Update axes
    for i in range(1, 5):
        fig.update_xaxes(title_text="Date", row=i, col=1, showgrid=True, gridwidth=1, dtick="M12", gridcolor='lightgray')
        fig.update_yaxes(
            title_text="Breadth (%)", 
            row=i, col=1, 
            range=[0, 100],
            showgrid=True, 
            gridwidth=1, 
            gridcolor='lightgray'
        )
    
    fig.update_layout(
        title_text='S&P 500 Equal-Weighted Market Breadth (% Stocks with Positive Momentum)',
        title_font_size=18,
        height=1400,
        width=1100,
        hovermode='x unified'
    )
    
    fig.write_html('sp500_equal_weighted_breadth.html')
    print("\nSaved interactive equal-weighted breadth plots to 'sp500_equal_weighted_breadth.html'")
    fig.show()

def plot_cap_weighted_history(history_df):
    """Plot historical cap-weighted breadth data"""
    if len(history_df) < 2:
        print("\nNot enough historical data to plot cap-weighted breadth yet.")
        print("Run this script multiple times to build up history.")
        return
    
    fig = go.Figure()
    
    colors = {'1M': 'steelblue', '3M': 'darkgreen', '6M': 'darkorange', '12M': 'crimson'}
    
    # Pivot data for plotting
    for window in MOMENTUM_WINDOWS.keys():
        window_data = history_df[history_df['Window'] == window].copy()
        window_data['Date'] = pd.to_datetime(window_data['Date'], format = "mixed")
        window_data = window_data.sort_values('Date')
        
        fig.add_trace(
            go.Scatter(
                x=window_data['Date'],
                y=window_data['Breadth_Pct'],
                name=window,
                mode='lines+markers',
                line=dict(color=colors.get(window, 'blue'), width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              '<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                              '<b>Breadth</b>: %{y:.2f}%<br>' +
                              '<extra></extra>'
            )
        )
    
    # 50% reference line
    fig.add_hline(
        y=50, line_dash="dash", line_color="red", line_width=1.5,
        opacity=0.7, annotation_text="50% Reference"
    )
    
    # Shaded regions
    fig.add_hrect(y0=0, y1=10, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=90, y1=100, fillcolor="green", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title='S&P 500 Market-Cap Weighted Breadth Over Time',
        title_font_size=18,
        xaxis_title='Date',
        yaxis_title='Breadth (% of Market Cap with Positive Momentum)',
        hovermode='x unified',
        height=500,
        width=1100,
        legend=dict(
            title='Momentum Window',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray',dtick="W1", tickangle = -45),
        yaxis=dict(range=[0, 100], showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    
    fig.write_html('sp500_cap_weighted_breadth_history.html')
    print("Saved interactive cap-weighted breadth history to 'sp500_cap_weighted_breadth_history.html'")
    fig.show()

def main():
    print("=" * 70)
    print("S&P 500 MARKET BREADTH CALCULATOR")
    print("=" * 70)
    
    # 1. Get S&P 500 components and weights
    components_df = scrape_sp500_components()
    symbols = components_df['symbol'].tolist()
    
    # 2. Download historical price data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_YEARS*365 + 365)  # Extra buffer
    
    prices_df = download_price_data(symbols, start_date, end_date)
    
    if prices_df.empty:
        print("ERROR: No price data downloaded. Exiting.")
        return
    
    # 3. Calculate equal-weighted breadth
    print("\n" + "=" * 70)
    print("CALCULATING EQUAL-WEIGHTED BREADTH")
    print("=" * 70)
    
    breadth_df = calculate_equal_weighted_breadth(prices_df)
    
    # Print summary statistics
    print("\n" + "-" * 70)
    print("EQUAL-WEIGHTED BREADTH SUMMARY (Last 5 Years)")
    print("-" * 70)
    recent_breadth = breadth_df.tail(252*5)  # Last 5 years
    summary = recent_breadth.describe().round(2)
    print(summary)
    
    print("\n" + "-" * 70)
    print("CURRENT EQUAL-WEIGHTED BREADTH")
    print("-" * 70)
    print(breadth_df.tail(1).to_string())
    
    # 4. Plot equal-weighted breadth
    plot_equal_weighted_breadth(breadth_df)
    
    # 5. Calculate cap-weighted breadth for latest day
    print("\n" + "=" * 70)
    print("CALCULATING MARKET-CAP WEIGHTED BREADTH (LATEST DAY)")
    print("=" * 70)
    
    cap_weighted_latest = calculate_cap_weighted_breadth(prices_df, components_df)
    
    print("\n" + "-" * 70)
    print("MARKET-CAP WEIGHTED BREADTH (CURRENT)")
    print("-" * 70)
    print(cap_weighted_latest.to_string(index=False))
    
    # 6. Save and plot cap-weighted history
    history_df = save_cap_weighted_data(cap_weighted_latest)
    
    # 7. Save weights history
    print("\n" + "=" * 70)
    print("SAVING S&P 500 WEIGHTS HISTORY")
    print("=" * 70)
    
    latest_date_str = cap_weighted_latest['Date'].iloc[0]
    weights_history = save_weights_history(components_df, latest_date_str)
    
    # 8. Plot cap-weighted history
    plot_cap_weighted_history(history_df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nFiles generated:")
    print(f"  1. sp500_equal_weighted_breadth.html - Interactive equal-weighted breadth charts")
    print(f"  2. {DATA_FILE} - Historical cap-weighted data")
    print(f"  3. {WEIGHTS_FILE} - Historical S&P 500 constituent weights")
    print(f"  4. sp500_cap_weighted_breadth_history.html - Interactive cap-weighted history chart")
    print(f"\nRun this script periodically to build up cap-weighted breadth history and track")
    print(f"S&P 500 constituent weight changes over time.")

if __name__ == "__main__":
    main()


# In[3]:


#Can we reasonably use the most recent cap weights to calculate the cap weighted breadth over the past 5 years? I don't think so because 
#look at NVDA - the mkt cap went from 200B to 4.5T from 2020 to 2026


