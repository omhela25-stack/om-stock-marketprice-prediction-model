# stockpredictor.app.py
# Merged & refactored: keeps your LSTM/LR + adds full analysis stack, multi-source data, indicators,
# model comparison, feature importance, richer charts, downloads, API status, and a fresh UI layout.
# Sources merged: your original stockpredictor.app.py and the provided app.py (features & flows).

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# ---- Optional deps (guarded) ----
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SK_OK = True
except Exception:
    SK_OK = False

# Keras (optional)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_OK = True
except Exception:
    TF_OK = False

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="Quantum Stocks Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- API setup (Alpha Vantage) ----
AV_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "RSMSPEMM4AQHT6QL")  # replace with your key in secrets
AV_BASE_URL = "https://www.alphavantage.co/query"

# ---- Style: new vibe (distinct from the reference app) ----
st.markdown("""
<style>
:root {
  --pri:#6e56cf; --pri2:#22c55e; --bg:#0b0d12; --panel:#121521; --muted:#9aa0b4; --line:#23263a;
}
html, body, [class*="css"]  { font-family: ui-sans-serif, Inter, system-ui; }
.stApp { background: radial-gradient(1200px 500px at 10% -10%, rgba(110,86,207,.15), transparent), var(--bg); }
.block-container { padding-top: 1.2rem; }
h1.title {
  font-size: 2.2rem; font-weight: 800; letter-spacing:.2px;
  background: linear-gradient(90deg, var(--pri), var(--pri2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: .3rem;
}
p.sub { color: var(--muted); margin-top:0; margin-bottom: 1.2rem; }
div.card {
  background: var(--panel); border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; 
}
.badge { display:inline-block; padding:.25rem .5rem; border-radius:999px; font-size:.8rem; border:1px solid var(--line); color:#cbd5e1; }
.stButton>button { background: linear-gradient(90deg, var(--pri), var(--pri2)); color:white; border:0; border-radius:9px; padding:.6rem 1rem; font-weight:700; }
.stButton>button:hover { opacity:.95; transform: translateY(-1px); }
hr.sep { border:0; height:1px; background: var(--line); margin: .8rem 0 1rem; }
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Quantum Stocks Lab</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub">Multi-model market analysis with unified data sources, technicals, and predictions.</p>', unsafe_allow_html=True)

# ------------------- Helper Data -------------------
RELIABLE_TICKERS = {
    "US": {
        "AAPL": "Apple", "GOOGL": "Alphabet", "MSFT": "Microsoft", "TSLA": "Tesla",
        "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta", "NFLX": "Netflix", "JPM": "JPMorgan", "V": "Visa"
    },
    "IN": {
        "RELIANCE.NSE": "Reliance", "TCS.NSE": "TCS", "INFY.NSE": "Infosys", "HDFCBANK.NSE": "HDFC Bank",
        "WIPRO.NSE": "Wipro", "ITC.NSE": "ITC", "SBIN.NSE": "SBI"
    }
}

def _period_to_days(p):
    return {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}.get(p,365)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_alpha_vantage(ticker, outputsize="full"):
    """Alpha Vantage TIME_SERIES_DAILY (adjusted if available)."""
    params = {
        "function":"TIME_SERIES_DAILY",
        "symbol":ticker,
        "apikey":AV_API_KEY,
        "outputsize":outputsize,
        "datatype":"json"
    }
    r = requests.get(AV_BASE_URL, params=params, timeout=30)
    data = r.json()
    if "Error Message" in data:
        raise RuntimeError(data["Error Message"])
    key = next((k for k in data.keys() if "Time Series" in k), None)
    if not key:
        raise RuntimeError("Alpha Vantage returned no time series.")
    df = pd.DataFrame.from_dict(data[key], orient="index")
    # AV daily keys are 1. open ... 5. volume
    df.columns = ["Open","High","Low","Close","Volume"][: len(df.columns)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().reset_index().rename(columns={"index":"Date"})
    df.attrs["source"] = "alpha_vantage"
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yfinance(ticker, period="5y"):
    if not YF_OK:
        raise RuntimeError("yfinance not available")
    hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError("yfinance returned empty data.")
    df = hist.reset_index()[["Date","Open","High","Low","Close","Volume"]]
    df.attrs["source"] = "yfinance"
    return df

def gen_sample_data(ticker, period):
    """Fallback synthetic series, stable yet realistic-ish."""
    days = _period_to_days(period)
    rng = pd.date_range(end=datetime.now(), periods=days, freq="B")
    base = 150 if "." not in ticker else 1200
    rs = np.random.RandomState(abs(hash(ticker)) % (2**32))
    rets = rs.normal(0.0003, 0.018, len(rng))
    price = [base]
    for r in rets[1:]:
        price.append(max(1, price[-1]*(1+r)))
    price = np.array(price)
    high = price*(1+rs.normal(0.004,0.01,len(price)))
    low  = price*(1-rs.normal(0.004,0.01,len(price)))
    openp= np.r_[price[0], price[:-1]*(1+rs.normal(0,0.003,len(price)-1))]
    vol  = rs.lognormal(np.log(800_000 if base<500 else 220_000), 0.8, len(price)).astype(int)
    df = pd.DataFrame({"Date":rng,"Open":openp,"High":high,"Low":low,"Close":price,"Volume":vol})
    df.attrs["source"] = "sample_data"
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_unified(ticker, period="1y"):
    """Unified fetch: Alpha Vantage ➜ yfinance ➜ sample."""
    # Normalize certain Indian suffixes to AV style used in provided reference
    if ticker.endswith(".BO") or ticker.endswith(".NS"):
        ticker = ticker.split(".")[0] + ".NSE"
    # AV first (matches merged reference) :contentReference[oaicite:2]{index=2}
    try:
        time.sleep(1)  # gentle on AV rate limits
        df = fetch_alpha_vantage(ticker, outputsize="full")
    except Exception:
        # yfinance fallback (if installed) :contentReference[oaicite:3]{index=3}
        try:
            yf_period = "5y" if period in {"2y","5y"} else "1y"
            df = fetch_yfinance(ticker, period=yf_period)
        except Exception:
            df = gen_sample_data(ticker, period)
    # Trim to period window
    start = datetime.now() - timedelta(days=_period_to_days(period))
    df = df[df["Date"] >= start].reset_index(drop=True)
    return df

def calc_rsi(close, window=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(window).mean()
    loss = (-d.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

def enrich(df):
    if df is None or df.empty: return df
    z = df.copy()
    z["MA20"] = z["Close"].rolling(20).mean()
    z["MA50"] = z["Close"].rolling(50).mean()
    z["RSI"]  = calc_rsi(z["Close"])
    z["Ret"]  = z["Close"].pct_change()
    z["Vol_MA10"] = z["Volume"].rolling(10).mean()
    for k in [1,2,3,5]:
        z[f"Close_lag{k}"] = z["Close"].shift(k)
    z = z.dropna().reset_index(drop=True)
    return z

def get_stock_meta(ticker):
    base = ticker.split(".")[0].upper()
    # simple meta to avoid copying reference verbatim; covers main cases :contentReference[oaicite:4]{index=4}
    m = {
        "AAPL":("Apple Inc.","Technology","USD"),
        "GOOGL":("Alphabet Inc.","Technology","USD"),
        "MSFT":("Microsoft","Technology","USD"),
        "TSLA":("Tesla","Consumer Cyclical","USD"),
        "RELIANCE":("Reliance Industries","Energy","INR"),
        "TCS":("Tata Consultancy Services","Technology","INR"),
        "INFY":("Infosys","Technology","INR"),
    }
    name, sector, ccy = m.get(base, (ticker,"Unknown","USD"))
    return {"name":name, "sector":sector, "currency":ccy, "market_cap":"N/A"}

# ------------------- ML Utilities -------------------
def split_scale(df):
    feats = ["Open","High","Low","Volume","MA20","MA50","RSI","Ret","Vol_MA10"] + [f"Close_lag{k}" for k in [1,2,3,5] if f"Close_lag{k}" in df.columns]
    feats = [c for c in feats if c in df.columns]
    X = df[feats].copy()
    y = df["Close"].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n = len(X)
    cut = int(n*0.8)
    return (Xs[:cut], Xs[cut:], y[:cut].values, y[cut:].values, scaler, feats)

def train_random_forest(df):
    if not SK_OK: return None
    Xtr, Xte, ytr, yte, scaler, feats = split_scale(df)
    model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    ytrp = model.predict(Xtr)
    ytep = model.predict(Xte)
    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(ytr, ytrp))),
        "test_rmse":  float(np.sqrt(mean_squared_error(yte, ytep))),
        "train_mae":  float(np.mean(np.abs(ytr - ytrp))),
        "test_mae":   float(np.mean(np.abs(yte - ytep))),
        "train_r2":   float(r2_score(ytr, ytrp)),
        "test_r2":    float(r2_score(yte, ytep)),
        "train_size": int(len(Xtr)),
        "test_size":  int(len(Xte)),
    }
    fi = pd.DataFrame({"feature":feats, "importance":model.feature_importances_}).sort_values("importance", ascending=False)
    return model, scaler, metrics, fi

def predict_next_close(model, scaler, df):
    feats = ["Open","High","Low","Volume","MA20","MA50","RSI","Ret","Vol_MA10"] + [f"Close_lag{k}" for k in [1,2,3,5] if f"Close_lag{k}" in df.columns]
    feats = [c for c in feats if c in df.columns]
    Xlast = df[feats].iloc[[-1]].values
    Xs = scaler.transform(Xlast)
    return float(model.predict(Xs)[0])

# ---- Your LSTM + Linear Regression (integrated & optional) :contentReference[oaicite:5]{index=5}
def make_lstm_dataset(arr, look_back=60):
    X, y = [], []
    for i in range(len(arr)-look_back):
        X.append(arr[i:i+look_back, 0])
        y.append(arr[i+look_back, 0])
    X = np.array(X); y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# ------------------- Sidebar -------------------
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Controls")
    mode = st.radio("Market", ["US", "IN", "Custom"], horizontal=True)
    if mode == "US":
        ticker = st.selectbox("Choose Stock", list(RELIABLE_TICKERS["US"].keys()))
        st.caption(f"Selected: {RELIABLE_TICKERS['US'][ticker]}")
    elif mode == "IN":
        ticker = st.selectbox("Choose Stock", list(RELIABLE_TICKERS["IN"].keys()))
        st.caption(f"Selected: {RELIABLE_TICKERS['IN'][ticker]}")
    else:
        ticker = st.text_input("Custom Ticker", "AAPL")

    period = st.selectbox("History Window", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
    pred_days = st.slider("Next-day focus (for signals)", 1, 10, 1)
    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    st.markdown("### 🔌 Status")
    yf_badge = "✅ yfinance ready" if YF_OK else "⚠️ yfinance missing"
    tf_badge = "✅ TensorFlow ready" if TF_OK else "⚠️ TensorFlow missing"
    sk_badge = "✅ scikit-learn ready" if SK_OK else "⚠️ scikit-learn missing"
    st.markdown(f'<span class="badge">{yf_badge}</span> <span class="badge">{sk_badge}</span> <span class="badge">{tf_badge}</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    run_btn = st.button("🚀 Analyze & Predict", use_container_width=True)

# ------------------- Action -------------------
if not run_btn:
    st.info("Pick a ticker and press **Analyze & Predict** to start.")
    st.stop()

with st.spinner("Pulling market data & building features..."):
    raw = fetch_unified(ticker, period=period)
    if raw is None or raw.empty:
        st.error("No data returned for this ticker/period.")
        st.stop()
    data = enrich(raw)

src = raw.attrs.get("source", "unknown")
meta = get_stock_meta(ticker)
ccy = meta["currency"]
ccy_sym = "$" if ccy == "USD" else "₹" if ccy == "INR" else ccy

st.success(f"Loaded **{len(raw)}** rows for **{ticker}** via **{src}**")
st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ------------------- Layout Tabs (renamed & reorganized) -------------------
tab_overview, tab_models, tab_charts, tab_importance, tab_table = st.tabs(
    ["Overview 📋", "Models 🤖", "Charts 📈", "Feature Impact 🎯", "Data 🔎"]
)

# ---- OVERVIEW ----
with tab_overview:
    colA, colB, colC, colD = st.columns(4)
    last_close = raw["Close"].iloc[-1]
    prev_close = raw["Close"].iloc[-2] if len(raw) > 1 else last_close
    delta = last_close - prev_close
    pct = (delta / prev_close * 100) if prev_close else 0.0
    vol = int(raw["Volume"].iloc[-1])

    with colA: st.metric("Last Close", f"{ccy_sym}{last_close:,.2f}", f"{pct:.2f}%")
    with colB: st.metric("Day Change", f"{ccy_sym}{delta:,.2f}")
    with colC: st.metric("Volume", f"{vol:,}")
    with colD: st.metric("Volatility (σ)", f"{raw['Close'].pct_change().std()*100:.2f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Company**")
        st.write(meta["name"])
        st.write(f"Sector: {meta['sector']}")
        st.write(f"Currency: {meta['currency']}")
    with c2:
        st.markdown("**Range & Stats**")
        st.write(f"52W High: {ccy_sym}{raw['High'].max():.2f}")
        st.write(f"52W Low : {ccy_sym}{raw['Low'].min():.2f}")
        st.write(f"Avg Vol : {raw['Volume'].mean():,.0f}")

# ---- MODELS ----
with tab_models:
    left, right = st.columns([1,1])

    # Random Forest (fast)
    with left:
        st.subheader("Random Forest (baseline)")
        if not SK_OK:
            st.warning("scikit-learn not available in this environment.")
        else:
            model, scaler, m, fi = train_random_forest(data)
            if model is None:
                st.error("Model training failed.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("R² (test)", f"{m['test_r2']:.3f}")
                col2.metric("RMSE (test)", f"{m['test_rmse']:.2f}")
                col3.metric("MAE (test)", f"{m['test_mae']:.2f}")

                # Next-day-style estimate (using last row features)
                try:
                    next_pred = predict_next_close(model, scaler, data)
                    cur = data["Close"].iloc[-1]
                    d = next_pred - cur
                    p = d/cur*100 if cur else 0
                    st.info(f"Next Est. Close: **{ccy_sym}{next_pred:.2f}**  (Δ {ccy_sym}{d:.2f}, {p:.2f}%)")
                    if p > 2: st.success("Signal: Bullish bias")
                    elif p > 0: st.info("Signal: Mild bullish")
                    elif p > -2: st.warning("Signal: Neutral")
                    else: st.error("Signal: Bearish bias")
                except Exception as e:
                    st.warning(f"Prediction step failed: {e}")

    # Your LSTM + Linear Regression (optional & quick)
    with right:
        st.subheader("Deep LSTM + Linear Regression (optional)")
        if not TF_OK or not SK_OK:
            st.info("Enable TensorFlow & scikit-learn in the environment to run these models.")
        else:
            look_back = st.slider("LSTM look-back window", 30, 120, 60, 10)
            epochs = st.slider("Train epochs", 1, 10, 5)
            run_deep = st.checkbox("Train & compare LSTM / Linear Regression", value=False)

            if run_deep:
                # Prepare adjusted-close-like series: we’ll use Close to avoid extra API calls (keeps experience smooth)
                series = raw[["Close"]].astype(float)
                scaler_mm = MinMaxScaler(feature_range=(0,1))
                scaled = scaler_mm.fit_transform(series)

                X, y = make_lstm_dataset(scaled, look_back)
                n = len(X); cut = int(n*0.8)
                Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

                # LSTM
                lstm = build_lstm((Xtr.shape[1], 1))
                lstm.fit(Xtr, ytr, epochs=epochs, batch_size=32, verbose=0)
                ypred_lstm = lstm.predict(Xte, verbose=0)
                ypred_lstm_inv = scaler_mm.inverse_transform(ypred_lstm)
                yte_inv = scaler_mm.inverse_transform(yte.reshape(-1,1))
                mse_lstm = mean_squared_error(yte_inv, ypred_lstm_inv)

                # Linear Regression baseline (simple time index)
                X_idx = np.arange(len(scaled)).reshape(-1,1)
                Xtr_lr, Xte_lr = X_idx[:cut+look_back], X_idx[cut+look_back:]
                ytr_lr, yte_lr = scaled[:cut+look_back], scaled[cut+look_back:]
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression().fit(Xtr_lr, ytr_lr)
                ypred_lr = lr.predict(Xte_lr)
                ypred_lr_inv = scaler_mm.inverse_transform(ypred_lr)
                yte_lr_inv = scaler_mm.inverse_transform(yte_lr)
                mse_lr = mean_squared_error(yte_lr_inv, ypred_lr_inv)

                cA, cB = st.columns(2)
                with cA:
                    st.metric("LSTM MSE", f"{mse_lstm:.4f}")
                    comp_df = pd.DataFrame({"Actual": yte_inv.flatten(), "LSTM": ypred_lstm_inv.flatten()})
                    st.line_chart(comp_df)
                with cB:
                    st.metric("Linear Regression MSE", f"{mse_lr:.4f}")
                    comp_lr_df = pd.DataFrame({"Actual": yte_lr_inv.flatten(), "LinearReg": ypred_lr_inv.flatten()})
                    st.line_chart(comp_lr_df)

# ---- CHARTS ----
with tab_charts:
    st.subheader(f"{ticker} — Price & Indicators")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw["Date"], y=raw["Close"], name="Close", line=dict(color="#6e56cf", width=2.5)))
    if "MA20" in data.columns:
        fig.add_trace(go.Scatter(x=data["Date"], y=data["MA20"], name="MA 20", line=dict(color="#22c55e", width=1.8, dash="dash")))
    if "MA50" in data.columns:
        fig.add_trace(go.Scatter(x=data["Date"], y=data["MA50"], name="MA 50", line=dict(color="#ef4444", width=1.8, dash="dot")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0d12", plot_bgcolor="#0b0d12",
                      xaxis_title="Date", yaxis_title=f"Price ({ccy_sym})", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fv = go.Figure()
        fv.add_trace(go.Bar(x=raw["Date"], y=raw["Volume"], name="Volume", marker_color="rgba(110,86,207,.6)"))
        fv.update_layout(template="plotly_dark", paper_bgcolor="#0b0d12", plot_bgcolor="#0b0d12",
                         xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fv, use_container_width=True)
    with c2:
        if "RSI" in data.columns and not data["RSI"].isna().all():
            fr = go.Figure()
            fr.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], name="RSI", line=dict(color="#22c55e", width=2.2)))
            fr.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought 70")
            fr.add_hline(y=30, line_dash="dash", line_color="#22c55e", annotation_text="Oversold 30")
            fr.update_layout(template="plotly_dark", paper_bgcolor="#0b0d12", plot_bgcolor="#0b0d12",
                             xaxis_title="Date", yaxis_title="RSI", yaxis=dict(range=[0,100]))
            st.plotly_chart(fr, use_container_width=True)

# ---- FEATURE IMPACT ----
with tab_importance:
    st.subheader("Model Feature Impact")
    if SK_OK:
        trained = train_random_forest(data)
        if trained is not None:
            _, _, metrics_tmp, fi_tmp = trained
            if fi_tmp is not None and not fi_tmp.empty:
                bar = px.bar(fi_tmp.head(12), x="importance", y="feature", orientation="h",
                             color="importance", color_continuous_scale="plasma",
                             title="Top Features Driving Predictions", template="plotly_dark")
                bar.update_layout(paper_bgcolor="#0b0d12", plot_bgcolor="#0b0d12", yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(bar, use_container_width=True)
                st.caption("Close lags, moving averages, RSI, and volume trends often rank highest for short-horizon signals.")
            col1, col2, col3 = st.columns(3)
            col1.metric("R² (test)", f"{metrics_tmp['test_r2']:.3f}")
            col2.metric("RMSE (test)", f"{metrics_tmp['test_rmse']:.2f}")
            col3.metric("MAE (test)", f"{metrics_tmp['test_mae']:.2f}")
    else:
        st.info("Install scikit-learn to view feature impact.")

# ---- DATA ----
with tab_table:
    st.subheader("Recent Data (last 60 rows)")
    view = data.tail(60).copy()
    view["Date"] = pd.to_datetime(view["Date"]).dt.strftime("%Y-%m-%d")
    cols = [c for c in ["Date","Open","High","Low","Close","Volume","MA20","MA50","RSI"] if c in view.columns]
    st.dataframe(view[cols], use_container_width=True, height=420)
    csv = raw.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name=f"{ticker}_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# ---- DISCLAIMER (styled) ----
st.markdown("""
<div class="card">
<strong>Disclaimer:</strong> This app is for research & education only. Markets are volatile; 
models can be wrong. Not financial advice. Always perform independent due diligence.
<br><span class="badge">Data sources: Alpha Vantage, yfinance (fallback), or synthetic sample when unavailable.</span>
</div>
""", unsafe_allow_html=True)

   

