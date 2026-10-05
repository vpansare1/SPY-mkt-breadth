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
    """Get S&P 500 components and calculate weights from market caps"""
    print("Getting S&P 500 components from Wikipedia...")
    
    try:
        # Get constituent list from Wikipedia with proper headers
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Use requests to get the HTML with headers, then parse with pandas
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        #parse html tables from the response
        from io import StringIO
        tables = pd.read_html(StringIO(response.text))
        sp500_df = tables[0]
        symbols = sp500_df['Symbol'].tolist()
        
        # Fix symbol formats for yfinance
        symbol_corrections = {
            'BRK.B': 'BRK-B',
            'BF.B': 'BF-B'
        }
        symbols = [symbol_corrections.get(s, s) for s in symbols]
        
        print(f"Found {len(symbols)} components from Wikipedia")
        
        # Calculate weights from market caps
        print("Calculating weights from current market caps...")
        market_caps = {}
        failed = []
        
        for i, symbol in enumerate(symbols, 1):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(symbols)}")
            
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_cap = info.get('marketCap', 0)
                if market_cap > 0:
                    market_caps[symbol] = market_cap
                else:
                    failed.append(symbol)
            except Exception as e:
                failed.append(symbol)
                continue
        
        if failed:
            print(f"  Could not get market cap for {len(failed)} symbols: {failed[:10]}...")
        
        # Calculate weights as percentage of total market cap
        total_cap = sum(market_caps.values())
        weights_dict = {symbol: (cap/total_cap)*100 for symbol, cap in market_caps.items()}
        
        df = pd.DataFrame({
            'symbol': list(weights_dict.keys()),
            'weight': list(weights_dict.values())
        })
        
        print(f"Calculated weights for {len(df)} components")
        print(f"Total weight: {df['weight'].sum():.2f}%")
        
        return df
        
    except Exception as e:
        print(f"Error getting components: {e}")
        print("Falling back to cached weights...")
        
        # Load from existing file if available
        if os.path.exists('sp500_weights_history.csv'):
            weights_df = pd.read_csv('sp500_weights_history.csv', index_col=0)
            latest_date = weights_df.columns[-1]
            latest_weights = weights_df[latest_date].dropna()
            df = pd.DataFrame({
                'symbol': latest_weights.index,
                'weight': latest_weights.values
            })
            print(f"Loaded {len(df)} components from cached weights ({latest_date})")
            return df
        else:
            raise Exception("Cannot get components and no cached weights available")

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

def calculate_cap_weighted_breadth(prices_df, weights_df, min_total_weight=90.0):
    """Calculate market-cap weighted breadth for most recent day.

    Skips any window where total_weight < min_total_weight, which indicates
    a partial data download (silent NaN fill from yfinance). Weights are
    normalized to sum to ~100, so 90.0 means 90% coverage.
    """
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
        
        # Data quality gate: if coverage is too low, the breadth number is meaningless
        if total_weight < min_total_weight:
            print(f"  SKIPPING {window_name} for {latest_date.strftime('%Y-%m-%d')}: "
                  f"total_weight={total_weight:.2f} below threshold {min_total_weight} "
                  f"(likely partial data download)")
            continue
        
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

MEDIAN_MOMENTUM_WINDOWS = ['1M', '12M']
MEDIAN_MIN_STOCKS = 50  # skip days with too few stocks to make a meaningful median

def download_index_prices(start_date, end_date, symbol='SPY'):
    """Download the cap-weighted index proxy (SPY) for comparison"""
    try:
        hist = yf.Ticker(symbol).history(start=start_date, end=end_date)
        if hist.empty:
            print(f"WARNING: no price data for {symbol}")
            return None
        return hist['Close']
    except Exception as e:
        print(f"WARNING: could not download {symbol}: {e}")
        return None

def calculate_median_momentum(prices_df, index_prices=None):
    """Cross-sectional median and quartiles of stock momentum, plus the index's own momentum.

    Returns a DataFrame with columns like '1M_median', '1M_p25', '1M_p75', '1M_index', '1M_spread'
    (spread = index momentum minus median stock momentum, in percentage points).
    """
    out = {}
    for window_name in MEDIAN_MOMENTUM_WINDOWS:
        window_days = MOMENTUM_WINDOWS[window_name]
        momentum = calculate_momentum(prices_df, window_days) * 100
        enough = momentum.notna().sum(axis=1) >= MEDIAN_MIN_STOCKS

        out[f'{window_name}_median'] = momentum.median(axis=1).where(enough)
        out[f'{window_name}_p25'] = momentum.quantile(0.25, axis=1).where(enough)
        out[f'{window_name}_p75'] = momentum.quantile(0.75, axis=1).where(enough)

        if index_prices is not None:
            idx = index_prices.reindex(prices_df.index)
            idx_mom = (idx / idx.shift(window_days) - 1) * 100
            out[f'{window_name}_index'] = idx_mom
            out[f'{window_name}_spread'] = idx_mom - out[f'{window_name}_median']

    df = pd.DataFrame(out)
    return df.dropna(how='all', subset=[f'{w}_median' for w in MEDIAN_MOMENTUM_WINDOWS])

def plot_median_momentum(median_df, index_label='SPY'):
    """Median stock momentum vs the cap-weighted index, one panel per window"""
    has_index = f'{MEDIAN_MOMENTUM_WINDOWS[0]}_index' in median_df.columns
    fig = make_subplots(
        rows=len(MEDIAN_MOMENTUM_WINDOWS), cols=1,
        subplot_titles=[f'{w} Momentum: Median Stock vs {index_label}' if has_index
                        else f'{w} Momentum: Median Stock' for w in MEDIAN_MOMENTUM_WINDOWS],
        vertical_spacing=0.10
    )

    for i, window in enumerate(MEDIAN_MOMENTUM_WINDOWS, 1):
        show_legend = (i == 1)

        # Interquartile band (25th-75th percentile of stocks)
        fig.add_trace(go.Scatter(
            x=median_df.index, y=median_df[f'{window}_p75'],
            line=dict(width=0), hoverinfo='skip', showlegend=False,
            legendgroup='iqr'
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=median_df.index, y=median_df[f'{window}_p25'],
            fill='tonexty', fillcolor='rgba(70,130,180,0.15)',
            line=dict(width=0), name='25th-75th pct of stocks',
            showlegend=show_legend, legendgroup='iqr',
            hovertemplate='25th pct: %{y:.1f}%<extra></extra>'
        ), row=i, col=1)

        # Median stock
        fig.add_trace(go.Scatter(
            x=median_df.index, y=median_df[f'{window}_median'],
            line=dict(color='steelblue', width=2), name='Median stock',
            showlegend=show_legend, legendgroup='median',
            hovertemplate='Median stock: %{y:.1f}%<extra></extra>'
        ), row=i, col=1)

        # Cap-weighted index
        if has_index:
            fig.add_trace(go.Scatter(
                x=median_df.index, y=median_df[f'{window}_index'],
                line=dict(color='crimson', width=1.5), name=f'{index_label} (cap-weighted)',
                showlegend=show_legend, legendgroup='index',
                hovertemplate=f'{index_label}: ' + '%{y:.1f}%<extra></extra>'
            ), row=i, col=1)

        fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, row=i, col=1)
        fig.update_xaxes(showgrid=True, gridcolor='lightgray', dtick='M12', row=i, col=1)
        fig.update_yaxes(title_text=f'{window} return (%)', showgrid=True,
                         gridcolor='lightgray', zeroline=False, row=i, col=1)

    fig.update_layout(
        title_text='S&P 500 Median Stock Momentum vs Cap-Weighted Index',
        title_font_size=18,
        height=900,
        width=1100,
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.03, xanchor='left', x=0)
    )

    fig.write_html('sp500_median_momentum.html')
    print("\nSaved interactive median momentum chart to 'sp500_median_momentum.html'")
    fig.show()

def plot_median_vs_index_spread(median_df, index_label='SPY'):
    """Spread = index momentum minus median stock momentum (percentage points).

    Positive = the cap-weighted index is beating the typical stock (narrow, concentrated leadership).
    Negative = the typical stock is beating the index (broad participation).
    """
    if f'{MEDIAN_MOMENTUM_WINDOWS[0]}_spread' not in median_df.columns:
        print(f"\nSkipping median vs {index_label} spread chart: no index data")
        return

    fig = make_subplots(
        rows=len(MEDIAN_MOMENTUM_WINDOWS), cols=1,
        subplot_titles=[f'{w} Spread: {index_label} minus Median Stock' for w in MEDIAN_MOMENTUM_WINDOWS],
        vertical_spacing=0.10
    )

    for i, window in enumerate(MEDIAN_MOMENTUM_WINDOWS, 1):
        spread = median_df[f'{window}_spread']
        fig.add_trace(go.Scatter(
            x=spread.index, y=spread.clip(lower=0), fill='tozeroy',
            fillcolor='rgba(220,20,60,0.20)', line=dict(width=0),
            hoverinfo='skip', showlegend=False
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=spread.index, y=spread.clip(upper=0), fill='tozeroy',
            fillcolor='rgba(46,139,87,0.20)', line=dict(width=0),
            hoverinfo='skip', showlegend=False
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=spread.index, y=spread, line=dict(color='black', width=1.2),
            name=f'{window} spread', showlegend=False,
            hovertemplate=f'{window} spread: ' + '%{y:+.1f} pp<extra></extra>'
        ), row=i, col=1)

        fig.add_hline(y=0, line_color='gray', line_width=1, row=i, col=1)
        fig.update_xaxes(showgrid=True, gridcolor='lightgray', dtick='M12', row=i, col=1)
        fig.update_yaxes(title_text='Spread (pp)', showgrid=True, gridcolor='lightgray',
                         zeroline=False, row=i, col=1)

    fig.update_layout(
        title_text=(f'{index_label} vs Median S&P 500 Stock: Momentum Spread'
                    '<br><sup>Red = index beating typical stock (concentrated) | '
                    'Green = typical stock beating index (broad)</sup>'),
        title_font_size=18,
        height=800,
        width=1100,
        hovermode='x unified',
        plot_bgcolor='white'
    )

    fig.write_html('sp500_median_vs_spy_spread.html')
    print("Saved interactive median vs index spread chart to 'sp500_median_vs_spy_spread.html'")
    fig.show()

def calculate_breadth_spread(breadth_df, cap_history_file=DATA_FILE):
    """Spread = cap-weighted breadth minus equal-weighted breadth, on dates with cap-weighted data.

    Returns a DataFrame indexed by date with one column per momentum window (percentage points).
    """
    if not os.path.exists(cap_history_file):
        return pd.DataFrame()

    cap = pd.read_csv(cap_history_file)
    cap['Date'] = pd.to_datetime(cap['Date'], format='mixed').dt.normalize()
    # Same date can appear twice under different string formats; keep the last one written
    cap = cap.drop_duplicates(subset=['Date', 'Window'], keep='last')
    cap_wide = cap.pivot(index='Date', columns='Window', values='Breadth_Pct')

    eq = breadth_df.copy()
    if eq.index.tz is not None:
        eq.index = eq.index.tz_localize(None)
    eq.index = eq.index.normalize()
    eq = eq[~eq.index.duplicated(keep='last')]

    common = cap_wide.index.intersection(eq.index)
    windows = [w for w in MOMENTUM_WINDOWS if w in cap_wide.columns and w in eq.columns]
    spread = cap_wide.loc[common, windows] - eq.loc[common, windows]
    return spread.sort_index()

def plot_breadth_spread(spread_df):
    """Cap-weighted minus equal-weighted breadth, all windows on one chart"""
    if spread_df.empty or len(spread_df) < 2:
        print("\nNot enough overlapping data to plot the cap- vs equal-weighted breadth spread yet.")
        return

    colors = {'1M': 'steelblue', '3M': 'darkgreen', '6M': 'darkorange', '12M': 'crimson'}
    fig = go.Figure()
    for window in spread_df.columns:
        fig.add_trace(go.Scatter(
            x=spread_df.index, y=spread_df[window], name=window,
            mode='lines+markers', line=dict(color=colors.get(window, 'gray'), width=2),
            marker=dict(size=5),
            hovertemplate=f'{window}: ' + '%{y:+.1f} pp<extra></extra>'
        ))

    fig.add_hline(y=0, line_color='gray', line_width=1)
    fig.update_layout(
        title=('S&P 500 Breadth Spread: Cap-Weighted minus Equal-Weighted'
               '<br><sup>Positive = large caps have broader positive momentum than the typical stock | '
               'Negative = large caps lagging</sup>'),
        title_font_size=18,
        xaxis_title='Date',
        yaxis_title='Spread (pp)',
        hovermode='x unified',
        height=550,
        width=1100,
        plot_bgcolor='white',
        legend=dict(title='Momentum Window', yanchor='top', y=0.99, xanchor='left', x=0.01),
        xaxis=dict(showgrid=True, gridcolor='lightgray', tickmode='auto', nticks=20, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False)
    )

    fig.write_html('sp500_breadth_spread.html')
    print("Saved interactive breadth spread chart to 'sp500_breadth_spread.html'")
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
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray',tickmode = "auto", nticks = 20, tickangle = -45),
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
    
    # Data quality diagnostics: non-NaN ticker counts for recent dates
    print("\n" + "-" * 70)
    print("DATA QUALITY: Non-NaN ticker counts for recent dates")
    print("-" * 70)
    total_tickers = prices_df.shape[1]
    recent_counts = prices_df.tail(10).notna().sum(axis=1)
    for date, count in recent_counts.items():
        pct = count / total_tickers * 100
        flag = "  <-- LOW COVERAGE" if pct < 90 else ""
        print(f"  {date.strftime('%Y-%m-%d')}: {count}/{total_tickers} ({pct:.1f}%){flag}")
    
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

    # 4b. Median stock momentum vs cap-weighted index
    print("\n" + "=" * 70)
    print("CALCULATING MEDIAN STOCK MOMENTUM")
    print("=" * 70)

    spy_prices = download_index_prices(start_date, end_date, 'SPY')
    median_df = calculate_median_momentum(prices_df, spy_prices)

    print("\n" + "-" * 70)
    print("CURRENT MEDIAN STOCK MOMENTUM (%)")
    print("-" * 70)
    print(median_df.tail(1).round(2).T.to_string())

    plot_median_momentum(median_df, 'SPY')
    plot_median_vs_index_spread(median_df, 'SPY')

    # 5. Calculate cap-weighted breadth for latest day
    print("\n" + "=" * 70)
    print("CALCULATING MARKET-CAP WEIGHTED BREADTH (LATEST DAY)")
    print("=" * 70)
    
    cap_weighted_latest = calculate_cap_weighted_breadth(prices_df, components_df)
    
    print("\n" + "-" * 70)
    print("MARKET-CAP WEIGHTED BREADTH (CURRENT)")
    print("-" * 70)
    
    if cap_weighted_latest.empty:
        print("No rows passed the data-quality threshold - skipping save for this run.")
    else:
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

    # 9. Cap-weighted minus equal-weighted breadth (uses the saved history, incl. today if saved)
    print("\n" + "=" * 70)
    print("CAP-WEIGHTED vs EQUAL-WEIGHTED BREADTH SPREAD")
    print("=" * 70)
    breadth_spread_df = calculate_breadth_spread(breadth_df)
    if not breadth_spread_df.empty:
        print(f"Overlapping dates: {len(breadth_spread_df)}")
        print("Latest spread (pp):")
        print(breadth_spread_df.tail(1).round(2).to_string())
    plot_breadth_spread(breadth_spread_df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nFiles generated:")
    print(f"  1. sp500_equal_weighted_breadth.html - Interactive equal-weighted breadth charts")
    print(f"  2. {DATA_FILE} - Historical cap-weighted data")
    print(f"  3. {WEIGHTS_FILE} - Historical S&P 500 constituent weights")
    print(f"  4. sp500_cap_weighted_breadth_history.html - Interactive cap-weighted history chart")
    print(f"  5. sp500_median_momentum.html - Median stock momentum vs SPY")
    print(f"  6. sp500_median_vs_spy_spread.html - SPY minus median stock momentum")
    print(f"  7. sp500_breadth_spread.html - Cap-weighted minus equal-weighted breadth")
    print(f"\nRun this script periodically to build up cap-weighted breadth history and track")
    print(f"S&P 500 constituent weight changes over time.")

if __name__ == "__main__":
    main()


# In[3]:


#Can we reasonably use the most recent cap weights to calculate the cap weighted breadth over the past 5 years? I don't think so because 
#look at NVDA - the mkt cap went from 200B to 4.5T from 2020 to 2026


