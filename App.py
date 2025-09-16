import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Helper: NSE Options Chain API
# -------------------------------
def get_option_chain(symbol="RELIANCE"):
    url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    response = session.get(url, headers=headers, timeout=10)
    data = response.json()
    return data

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="AI Options & Futures Predictor", layout="wide")
st.title("📊 AI Options & Futures Prediction App (NSE)")

# -------------------------------
# User Input
# -------------------------------
ticker = st.text_input("Enter NSE Stock/Index Symbol (e.g., RELIANCE, INFY, NIFTY):", "RELIANCE")
period = st.selectbox("Select Historical Period:", ["1mo", "3mo", "6mo", "1y"])
interval = st.selectbox("Select Interval:", ["15m", "1h", "1d"])

if st.button("🚀 Fetch & Predict"):
    try:
        # -------------------------------
        # Stock Data Fetch
        # -------------------------------
        st.write(f"Fetching historical data for {ticker}...")
        df = yf.download(f"{ticker}.NS", period=period, interval=interval)

        if df.empty:
            st.error("No data found. Try different symbol.")
        else:
            st.success("✅ Data fetched successfully!")
            st.line_chart(df["Close"])

            # -------------------------------
            # Feature Engineering
            # -------------------------------
            df["Return"] = df["Close"].pct_change()
            df["SMA_5"] = df["Close"].rolling(5).mean()
            df["SMA_10"] = df["Close"].rolling(10).mean()
            df["Volatility"] = df["Return"].rolling(5).std()
            df = df.dropna()

            df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
            features = ["Return", "SMA_5", "SMA_10", "Volatility"]
            X = df[features]
            y = df["Target"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            split = int(len(df) * 0.8)
            X_train, X_test = X_scaled[:split], X_scaled[split:]
            y_train, y_test = y[:split], y[split:]

            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

            acc = model.score(X_test, y_test)
            st.write(f"📈 Model Accuracy: **{acc*100:.2f}%**")

            latest_features = X_scaled[-1].reshape(1, -1)
            pred = model.predict(latest_features)[0]

            if pred == 1:
                st.success("📈 AI Prediction: Bullish trend expected ✅")
            else:
                st.error("📉 AI Prediction: Bearish trend expected ❌")

            # -------------------------------
            # Options Chain Data
            # -------------------------------
            st.subheader("📊 NSE Options Chain Data")
            try:
                option_data = get_option_chain(ticker.upper())
                records = option_data["records"]["data"]

                calls, puts = [], []
                for r in records:
                    ce = r.get("CE")
                    pe = r.get("PE")
                    if ce:
                        calls.append({
                            "Strike": ce["strikePrice"],
                            "LTP": ce["lastPrice"],
                            "OI": ce["openInterest"],
                            "IV": ce["impliedVolatility"]
                        })
                    if pe:
                        puts.append({
                            "Strike": pe["strikePrice"],
                            "LTP": pe["lastPrice"],
                            "OI": pe["openInterest"],
                            "IV": pe["impliedVolatility"]
                        })

                calls_df = pd.DataFrame(calls)
                puts_df = pd.DataFrame(puts)

                c1, c2 = st.columns(2)
                with c1:
                    st.write("### 📞 Calls Data")
                    st.dataframe(calls_df.head(10))
                with c2:
                    st.write("### 📉 Puts Data")
                    st.dataframe(puts_df.head(10))

                # -------------------------------
                # Simple Strategy Suggestion
                # -------------------------------
                st.subheader("🧠 Suggested Option Strategy")
                total_call_oi = calls_df["OI"].sum()
                total_put_oi = puts_df["OI"].sum()

                if pred == 1 and total_call_oi > total_put_oi:
                    st.success("👉 Strategy: **Bull Call Spread / Long Futures**")
                elif pred == 0 and total_put_oi > total_call_oi:
                    st.error("👉 Strategy: **Bear Put Spread / Short Futures**")
                else:
                    st.info("👉 Strategy: **Neutral Market → Iron Condor / Straddle**")

            except Exception as e:
                st.warning("⚠️ NSE options data could not be fetched. NSE blocks direct calls sometimes.")
                st.write(str(e))

            # -------------------------------
            # Data Preview
            # -------------------------------
            st.subheader("Recent Data")
            st.dataframe(df.tail(10))

    except Exception as e:
        st.error("❌ Error occurred.")
        st.write(str(e))
