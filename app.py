import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import datetime
import warnings

warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(page_title="Pro AI Stock Analyst", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-size: 16px;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] { 
        font-size: 22px !important; 
        white-space: normal !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.title("🏦 AI-Powered Financial Terminal")
    st.markdown("Advanced quantitative analysis, institutional moving averages, and Machine Learning projections.")
with col_header2:
    st.caption(f"Live Session: {datetime.datetime.now().strftime('%Y-%m-%d')}")
st.divider()

# 2. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Terminal Controls")
    ticker = st.text_input("Asset Ticker (e.g., AAPL, NVDA):", value="AAPL").upper()
    period_options = {"1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    selected_period = st.selectbox("Historical Data Range:", list(period_options.keys()), index=2)
    analyze_button = st.button("Execute AI Analysis", use_container_width=True, type="primary")

if analyze_button or ticker:
    with st.spinner(f"Aggregating market data and training AI for {ticker}..."):
        stock = yf.Ticker(ticker)
        df = stock.history(period=period_options[selected_period])
        
        try:
            company_name = stock.info.get('shortName', ticker)
        except:
            company_name = ticker
        
        if df.empty:
            st.error(f"Error: Unable to fetch data for '{ticker}'. Please verify the ticker symbol.")
        else:
            # 3. Calculate Indicators
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1/50, min_periods=50).mean()
            avg_loss = loss.ewm(alpha=1/50, min_periods=50).mean()
            rs = avg_gain / avg_loss
            df['RSI_50'] = 100 - (100 / (1 + rs))
            
            df = df.dropna()
            
            # 4. AI Machine Learning Engine
            df['Days'] = np.arange(len(df))
            X = df[['Days']]
            y = df['Close']
            model = LinearRegression()
            model.fit(X, y)
            
            future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
            predictions = model.predict(future_days)
            future_dates = [df.index[-1] + datetime.timedelta(days=i) for i in range(1, 31)]
            
            # Extract current metrics
            curr_price = df['Close'].iloc[-1]
            pred_price = predictions[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            sma_200 = df['SMA_200'].iloc[-1]
            ai_change = ((pred_price - curr_price) / curr_price) * 100
            cross_status = "GOLDEN CROSS 🟢" if sma_50 > sma_200 else "DEATH CROSS 🔴"

            st.subheader(f"Asset Overview: {company_name} ({ticker})")

            # 5. UI Cards for Metrics
            m1, m2, m3, m4 = st.columns([1, 1, 1.2, 1])
            with m1.container(border=True):
                st.metric("Current Asset Price", f"${curr_price:.2f}")
            with m2.container(border=True):
                st.metric("AI 30-Day Target", f"${pred_price:.2f}", f"{ai_change:.2f}% Projected")
            with m3.container(border=True):
                st.metric("Macro Structure", cross_status)
            with m4.container(border=True):
                st.metric("50-Day RSI Momentum", f"{df['RSI_50'].iloc[-1]:.2f}")

            # 6. Tabbed Workspace
            tab1, tab2, tab3 = st.tabs(["📈 Interactive Chart", "🤖 Detailed AI Report", "📁 Raw Dataset"])
            
            with tab1:
                # Chart separation increased to 0.15
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.15, row_heights=[0.75, 0.25])
                
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                             low=df['Low'], close=df['Close'], name='Market Price'), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='50-Day SMA', line=dict(color='blue', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='200-Day SMA', line=dict(color='red', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='AI Forecast', line=dict(color='yellow', dash='dash', width=2)), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI_50'], name='50-Day RSI', line=dict(color='#B794F4'), fill='tozeroy'), row=2, col=1)
                fig.add_hline(y=60, line_dash="dot", line_color="red", row=2, col=1)
                fig.add_hline(y=40, line_dash="dot", line_color="green", row=2, col=1)
                
                # Explicit axis titles added
                fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
                fig.update_yaxes(title_text="RSI Value", row=2, col=1)
                fig.update_xaxes(title_text="Date", row=2, col=1)
                
                # Top margin increased (t=60) to prevent modebar overlap
                fig.update_layout(height=750, margin=dict(l=40, r=40, t=60, b=40), template="plotly_dark", xaxis_rangeslider_visible=False)
                
                # Streamlit chart render config updated to manage the modebar
                st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

            with tab2:
                st.markdown("### 1. Technical Framework")
                if sma_50 > sma_200:
                    st.success(f"**Bullish Macro Trend:** The asset is currently in a {cross_status}. The short-term 50-day average (${sma_50:.2f}) has crossed above the long-term 200-day average (${sma_200:.2f}), indicating sustained upward momentum.")
                else:
                    st.error(f"**Bearish Macro Trend:** The asset is currently in a {cross_status}. The short-term 50-day average (${sma_50:.2f}) has crossed below the long-term 200-day average (${sma_200:.2f}), indicating sustained downward pressure.")
                
                st.markdown("### 2. Machine Learning Outlook")
                trend_direction = "appreciate" if pred_price > curr_price else "depreciate"
                st.info(f"**30-Day Forecast:** The linear regression AI model anticipates the asset will **{trend_direction}** over the next 30 trading days, reaching a target price of **${pred_price:.2f}** (a change of {ai_change:+.2f}%).")
                
                st.markdown("### 3. Momentum (RSI)")
                rsi_val = df['RSI_50'].iloc[-1]
                if rsi_val > 60:
                    st.warning(f"**Overbought:** The 50-day RSI is currently elevated at {rsi_val:.2f}. The asset may be slightly overvalued in the short term.")
                elif rsi_val < 40:
                    st.success(f"**Oversold:** The 50-day RSI is depressed at {rsi_val:.2f}. The asset may be undervalued, presenting a potential buying opportunity.")
                else:
                    st.write(f"**Neutral:** The 50-day RSI is balanced at {rsi_val:.2f}, indicating stable momentum without extreme volatility.")

            with tab3:
                st.subheader(f"Historical & Processed Dataset: {ticker}")
                st.dataframe(df.tail(100), use_container_width=True)
                st.download_button(label="Download CSV", data=df.to_csv().encode('utf-8'), file_name=f"{ticker}_data.csv", mime='text/csv')
