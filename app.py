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
st.set_page_config(page_title="AI Stock Predictor", layout="wide", page_icon="📈")
st.title("📈 AI-Based Stock Market Analysis & Prediction System")
st.markdown("Analyze historical data, institutional moving averages, volatility, and AI-driven price projections.")

# 2. Sidebar Input
st.sidebar.header("System Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker:", value="NVDA").upper()
analyze_button = st.sidebar.button("Run AI Analysis")

if analyze_button or ticker:
    with st.spinner(f"Fetching data and training AI for {ticker}..."):
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y")
        
        if df.empty:
            st.error("Invalid ticker symbol. Please try again.")
        else:
            # 3. Technical Indicators
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1/50, min_periods=50).mean()
            avg_loss = loss.ewm(alpha=1/50, min_periods=50).mean()
            rs = avg_gain / avg_loss
            df['RSI_50'] = 100 - (100 / (1 + rs))
            
            df['BB_Mid'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
            df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
            
            df = df.dropna()
            
            # 4. AI ML Model
            df['Days'] = np.arange(len(df))
            X = df[['Days']]
            y = df['Close']
            model = LinearRegression()
            model.fit(X, y)
            
            future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
            predictions = model.predict(future_days)
            future_dates = [df.index[-1] + datetime.timedelta(days=i) for i in range(1, 31)]
            
            # 5. Interactive Plotly Visualization
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # Top Chart: Price & MAs
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='white', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='50-Day SMA', line=dict(color='blue', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='200-Day SMA', line=dict(color='red', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='AI Forecast', line=dict(color='green', dash='dash', width=2)), row=1, col=1)
            
            # Bottom Chart: RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI_50'], name='50-Day RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=60, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=40, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(height=700, title_text=f"Interactive Technical & AI Analysis for {ticker}", template="plotly_dark")
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 6. Metrics Dashboard
            st.subheader("📊 Institutional-Grade AI Report")
            col1, col2, col3, col4 = st.columns(4)
            
            curr_price = df['Close'].iloc[-1]
            pred_price = predictions[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            sma_200 = df['SMA_200'].iloc[-1]
            
            ai_change = ((pred_price - curr_price) / curr_price) * 100
            cross = "GOLDEN CROSS 🟢" if sma_50 > sma_200 else "DEATH CROSS 🔴"
            
            col1.metric("Current Price", f"${curr_price:.2f}")
            col2.metric("AI 30-Day Target", f"${pred_price:.2f}", f"{ai_change:.2f}%")
            col3.metric("Macro Trend", cross)
            col4.metric("50-Day RSI", f"{df['RSI_50'].iloc[-1]:.2f}")
            
            st.info("The interactive chart above allows zooming, panning, and hovering for exact daily values. The AI model projects linear momentum based on historical structures.")
