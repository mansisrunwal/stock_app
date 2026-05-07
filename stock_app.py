import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 StockSense | Mansi & Husband", layout="wide",
                   page_icon="📈", initial_sidebar_state="expanded")

TICKERS = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'GOOGL': 'Alphabet',
    'AMZN': 'Amazon', 'META': 'Meta', 'TSLA': 'Tesla', 'JPM': 'JP Morgan',
    'V': 'Visa', 'AVGO': 'Broadcom', 'LLY': 'Eli Lilly', 'MRK': 'Merck',
    'COST': 'Costco', 'BAC': 'Bank of America', 'AMD': 'AMD',
    'NFLX': 'Netflix', 'ADBE': 'Adobe', 'CRM': 'Salesforce',
    'WMT': 'Walmart', 'HD': 'Home Depot', 'MA': 'Mastercard',
    'XOM': 'ExxonMobil', 'PG': 'P&G', 'KO': 'Coca-Cola',
    'JNJ': 'Johnson & Johnson', 'PEP': 'PepsiCo', 'MCD': "McDonald's",
    'ABBV': 'AbbVie', 'BRK-B': 'Berkshire', 'UNH': 'UnitedHealth'
}

USD_GBP = 0.79  # fallback; will try to fetch live

# ── HELPERS ─────────────────────────────────────────────────────────────────

def get_usd_gbp():
    try:
        fx = yf.download("GBPUSD=X", period="5d", interval="1d", progress=False, auto_adjust=True)
        return round(1 / float(fx['Close'].iloc[-1]), 4)
    except:
        return USD_GBP

def compute_indicators(df):
    df = df.copy()
    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()

    # Moving Averages
    df['SMA_20'] = close.rolling(20).mean()
    df['SMA_50'] = close.rolling(50).mean()
    df['EMA_12'] = close.ewm(span=12).mean()
    df['EMA_26'] = close.ewm(span=26).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_mid'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * bb_std
    df['BB_lower'] = df['BB_mid'] - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
    df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # Momentum
    df['Momentum_5'] = close.pct_change(5)
    df['Momentum_20'] = close.pct_change(20)
    df['Volatility'] = close.rolling(20).std() / close.rolling(20).mean()

    # Volume trend
    df['Volume_ratio'] = volume / volume.rolling(20).mean()

    # Price vs MA
    df['Price_vs_SMA20'] = (close - df['SMA_20']) / df['SMA_20']
    df['Price_vs_SMA50'] = (close - df['SMA_50']) / df['SMA_50']

    # Target: will price be higher in N days?
    df['Target_5d'] = (close.shift(-5) > close * 1.02).astype(int)
    df['Target_20d'] = (close.shift(-20) > close * 1.05).astype(int)
    df['Target_60d'] = (close.shift(-60) > close * 1.10).astype(int)

    return df.dropna()

FEATURES = ['RSI', 'MACD', 'MACD_hist', 'BB_position', 'BB_width',
            'Momentum_5', 'Momentum_20', 'Volatility', 'Volume_ratio',
            'Price_vs_SMA20', 'Price_vs_SMA50']

@st.cache_data(ttl=3600)
def fetch_and_train(ticker):
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df is None or len(df) < 150:
        return None
    df = compute_indicators(df)

    results = {}
    for horizon, target_col in [('Short (Days)', 'Target_5d'),
                                  ('Medium (Months)', 'Target_20d'),
                                  ('Long (Years)', 'Target_60d')]:
        X = df[FEATURES].values
        y = df[target_col].values
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[valid], y[valid]
        if len(X) < 80:
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        latest = scaler.transform(df[FEATURES].iloc[[-1]].values)
        prob = model.predict_proba(latest)[0][1]
        results[horizon] = {'probability': round(prob, 3), 'accuracy': round(acc, 3)}

    close = df['Close'].squeeze()
    latest_row = df.iloc[-1]
    current_price = float(close.iloc[-1])
    results['meta'] = {
        'current_price_usd': current_price,
        'rsi': float(latest_row['RSI']),
        'macd': float(latest_row['MACD']),
        'macd_hist': float(latest_row['MACD_hist']),
        'bb_position': float(latest_row['BB_position']),
        'momentum_5': float(latest_row['Momentum_5']),
        'momentum_20': float(latest_row['Momentum_20']),
        'sma20': float(latest_row['SMA_20']),
        'sma50': float(latest_row['SMA_50']),
        'volume_ratio': float(latest_row['Volume_ratio']),
        'price_history': close.tail(90).tolist(),
        'price_dates': [str(d.date()) for d in close.tail(90).index],
    }
    return results

def build_reasoning(meta, short_p, med_p, long_p):
    reasons = []
    rsi = meta['rsi']
    if rsi < 35:
        reasons.append(f"📉 RSI at {rsi:.1f} — oversold, likely bounce ahead")
    elif rsi > 65:
        reasons.append(f"📈 RSI at {rsi:.1f} — strong momentum but watch for pullback")
    else:
        reasons.append(f"⚖️ RSI at {rsi:.1f} — neutral, stable momentum")

    if meta['macd_hist'] > 0:
        reasons.append("✅ MACD histogram positive — bullish crossover signal")
    else:
        reasons.append("⚠️ MACD histogram negative — bearish pressure, wait for reversal")

    bb = meta['bb_position']
    if bb < 0.25:
        reasons.append("🟢 Price near Bollinger lower band — potential buy zone")
    elif bb > 0.75:
        reasons.append("🔴 Price near Bollinger upper band — overbought, consider waiting")
    else:
        reasons.append("🟡 Price mid-Bollinger — consolidation phase")

    if meta['momentum_20'] > 0.05:
        reasons.append(f"🚀 20-day momentum +{meta['momentum_20']*100:.1f}% — strong trend")
    elif meta['momentum_20'] < -0.05:
        reasons.append(f"📉 20-day momentum {meta['momentum_20']*100:.1f}% — downtrend caution")

    if meta['volume_ratio'] > 1.3:
        reasons.append("📊 Volume 30%+ above average — institutional activity detected")

    avg_prob = (short_p + med_p + long_p) / 3
    if avg_prob > 0.65:
        reasons.append(f"🤖 ML model confidence: {avg_prob*100:.0f}% — high conviction buy")
    elif avg_prob > 0.5:
        reasons.append(f"🤖 ML model confidence: {avg_prob*100:.0f}% — moderate opportunity")
    else:
        reasons.append(f"🤖 ML model confidence: {avg_prob*100:.0f}% — low conviction, risky")

    return reasons

def get_signal(prob):
    if prob >= 0.65: return "🟢 BUY", "green"
    if prob >= 0.5: return "🟡 HOLD", "orange"
    return "🔴 SELL/AVOID", "red"

def tier(price_gbp):
    if price_gbp < 30: return "🔵 Low Value"
    if price_gbp < 150: return "🟠 Mid Value"
    return "🔴 High Value"

# ── UI ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.big-title { font-size: 2.5rem; font-weight: 800; color: #1f77b4; }
.sub { font-size: 1rem; color: #888; }
.metric-card { background: #1e1e2e; border-radius: 12px; padding: 16px;
               border-left: 4px solid #1f77b4; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📈 StockSense</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">AI-powered stock intelligence for London investors · Prices in GBP · Updated live</div>',
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Flag_of_the_United_Kingdom.svg/320px-Flag_of_the_United_Kingdom.svg.png", width=80)
    st.title("⚙️ Settings")
    selected_tickers = st.multiselect("Select stocks to analyse", list(TICKERS.keys()),
                                       default=list(TICKERS.keys())[:15])
    horizon_filter = st.selectbox("Primary horizon", ['Short (Days)', 'Medium (Months)', 'Long (Years)'])
    tier_filter = st.multiselect("Value tier", ['🔵 Low Value', '🟠 Mid Value', '🔴 High Value'],
                                  default=['🔵 Low Value', '🟠 Mid Value', '🔴 High Value'])
    min_confidence = st.slider("Min ML confidence (%)", 40, 90, 50)
    st.markdown("---")
    st.markdown("💷 **Currency:** GBP (£)")
    st.markdown(f"🕒 **Last refresh:** {datetime.now().strftime('%d %b %Y %H:%M')}")
    refresh = st.button("🔄 Refresh Data Now")
    if refresh:
        st.cache_data.clear()
        st.rerun()

# Fetch FX
usd_gbp = get_usd_gbp()
st.caption(f"💱 Live rate: 1 USD = £{usd_gbp:.4f}")

# Main analysis
st.header("🏆 Top 10 Stock Recommendations")
st.info("🤖 Models trained on 2 years of daily data using Gradient Boosting. Re-trains automatically on each refresh.")

results_list = []
progress = st.progress(0, text="Fetching & training models...")

for i, ticker in enumerate(selected_tickers):
    progress.progress((i + 1) / len(selected_tickers), text=f"Analysing {ticker}...")
    res = fetch_and_train(ticker)
    if res and 'meta' in res:
        meta = res['meta']
        price_gbp = meta['current_price_usd'] * usd_gbp
        short_p = res.get('Short (Days)', {}).get('probability', 0.5)
        med_p = res.get('Medium (Months)', {}).get('probability', 0.5)
        long_p = res.get('Long (Years)', {}).get('probability', 0.5)
        horizon_prob = res.get(horizon_filter, {}).get('probability', 0.5)
        stock_tier = tier(price_gbp)
        signal, _ = get_signal(horizon_prob)
        if stock_tier in tier_filter and horizon_prob * 100 >= min_confidence:
            results_list.append({
                'Ticker': ticker,
                'Company': TICKERS[ticker],
                'Price (£)': round(price_gbp, 2),
                'Tier': stock_tier,
                'Short Score': short_p,
                'Med Score': med_p,
                'Long Score': long_p,
                'Horizon Score': horizon_prob,
                'Signal': signal,
                'RSI': round(meta['rsi'], 1),
                'meta': meta,
                'reasoning': build_reasoning(meta, short_p, med_p, long_p)
            })

progress.empty()

if not results_list:
    st.warning("No stocks matched your filters. Try lowering minimum confidence or broadening tier selection.")
    st.stop()

# Sort by horizon score and take top 10
results_df = pd.DataFrame(results_list).sort_values('Horizon Score', ascending=False).head(10).reset_index(drop=True)
results_df.index = results_df.index + 1  # rank from 1

# Summary table
st.subheader(f"📊 Rankings — {horizon_filter} horizon")
display_df = results_df[['Company', 'Ticker', 'Price (£)', 'Tier', 'Signal', 'RSI',
                           'Short Score', 'Med Score', 'Long Score']].copy()
display_df['Short Score'] = display_df['Short Score'].apply(lambda x: f"{x*100:.0f}%")
display_df['Med Score'] = display_df['Med Score'].apply(lambda x: f"{x*100:.0f}%")
display_df['Long Score'] = display_df['Long Score'].apply(lambda x: f"{x*100:.0f}%")
st.dataframe(display_df, use_container_width=True, height=420)

# Confidence chart
fig_bar = px.bar(results_df, x='Ticker', y='Horizon Score',
                  color='Horizon Score', color_continuous_scale='RdYlGn',
                  title=f"ML Confidence — {horizon_filter}",
                  labels={'Horizon Score': 'Buy Probability', 'Ticker': 'Stock'})
fig_bar.add_hline(y=0.65, line_dash="dash", line_color="green", annotation_text="Strong Buy threshold")
fig_bar.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Neutral")
st.plotly_chart(fig_bar, use_container_width=True)

# Detailed cards
st.markdown("---")
st.header("🔍 Detailed Analysis — Top 10")
for idx, row in results_df.iterrows():
    with st.expander(f"#{idx} · {row['Company']} ({row['Ticker']}) · £{row['Price (£)']} · {row['Signal']} · {row['Tier']}"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"£{row['Price (£)']}")
        col2.metric("Short-term", f"{row['Short Score']}")
        col3.metric("Mid-term", f"{row['Med Score']}")
        col4.metric("Long-term", f"{row['Long Score']}")

        # Price chart
        meta = row['meta']
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=meta['price_dates'], y=[p * usd_gbp for p in meta['price_history']],
            mode='lines', name='Price (£)', fill='tozeroy', line=dict(color='#1f77b4', width=2)
        ))
        fig_line.update_layout(title=f"{row['Ticker']} — Last 90 days (GBP)",
                                 xaxis_title="Date", yaxis_title="Price (£)",
                                 height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_line, use_container_width=True)

        # Reasoning
        st.subheader("🧠 Why this stock?")
        for reason in row['reasoning']:
            st.markdown(f"- {reason}")

        # 3-horizon bar
        horizon_data = pd.DataFrame({
            'Horizon': ['Short (Days)', 'Medium (Months)', 'Long (Years)'],
            'Buy Probability': [row['Short Score'], row['Med Score'], row['Long Score']]
        })
        fig_h = px.bar(horizon_data, x='Horizon', y='Buy Probability', color='Buy Probability',
                        color_continuous_scale='RdYlGn', range_y=[0, 1],
                        title="Buy Probability by Time Horizon")
        fig_h.add_hline(y=0.65, line_dash="dash", line_color="green")
        st.plotly_chart(fig_h, use_container_width=True)

# Portfolio simulator
st.markdown("---")
st.header("💰 Quick Portfolio Simulator")
st.caption("See how much to allocate across top picks")
budget = st.number_input("Your investment budget (£)", min_value=100, max_value=500000,
                          value=1000, step=100)
top5 = results_df.head(5)
alloc = budget / len(top5)
sim_data = []
for _, row in top5.iterrows():
    shares = alloc / row['Price (£)']
    sim_data.append({'Stock': row['Company'], 'Allocation (£)': round(alloc, 2),
                      'Shares': round(shares, 4), 'Price (£)': row['Price (£)']})
st.dataframe(pd.DataFrame(sim_data), use_container_width=True)

st.markdown("---")
st.caption("⚠️ **Disclaimer:** This app is for informational purposes only and does not constitute financial advice. "
           "Always do your own research before investing. Past performance does not guarantee future results.")
