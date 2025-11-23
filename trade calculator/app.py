import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import time
import requests
import random

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_headers():
    """Returns a random browser header to avoid bot detection."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0'
    ]
    return {'User-Agent': random.choice(user_agents)}

@st.cache_data(ttl=86400) # Cache S&P list for 24 hours
def get_sp500_tickers():
    """Scrapes the S&P 500 list from Wikipedia."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        return df['Symbol'].tolist()
    except Exception as e:
        return []

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    # Parse and sort
    dt_objs = []
    for d in dates:
        try:
            dt_objs.append(datetime.strptime(d, "%Y-%m-%d").date())
        except:
            continue # Skip weird formats
            
    sorted_dates = sorted(dt_objs)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            # Return the range up to this date
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]  
            break
    
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date >45 days found")

def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    if len(price_data) < window:
        return 0 # Not enough data
        
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    
    log_oc_sq = log_oc**2
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)) )
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return result.iloc[-1] if not result.empty else 0
    else:
        return result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)
    
    # Remove duplicates or NaNs
    valid_mask = ~np.isnan(days) & ~np.isnan(ivs)
    days = days[valid_mask]
    ivs = ivs[valid_mask]
    
    if len(days) < 2:
        return lambda x: ivs[0] if len(ivs) > 0 else 0

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]: return ivs[0]
        elif dte > days[-1]: return ivs[-1]
        else: return float(spline(dte))
    return term_spline

# ==========================================
# CORE LOGIC (Robust)
# ==========================================

def get_options_with_retry(stock, exp_date, retries=3):
    """Retries fetching option chain if it fails."""
    for i in range(retries):
        try:
            # Sleep increasingly longer if we fail
            if i > 0: time.sleep(1 * (i+1)) 
            return stock.option_chain(exp_date)
        except Exception as e:
            if i == retries - 1: raise e # Raise on last try
            pass

def compute_recommendation(ticker_symbol, progress_callback=None):
    try:
        ticker_symbol = ticker_symbol.strip().upper()
        
        # 1. INITIALIZATION
        # We do NOT use a custom session object here because yfinance 
        # has internal logic to fetch cookies/crumbs that custom sessions often break.
        stock = yf.Ticker(ticker_symbol)
        
        # 2. FETCH EARNINGS
        next_earnings = "N/A"
        try:
            cal = stock.calendar
            if cal and isinstance(cal, dict) and 'Earnings Date' in cal:
                dates = cal['Earnings Date']
                now = datetime.now().date()
                future_dates = [d for d in dates if d >= now]
                if future_dates:
                    next_earnings = future_dates[0].strftime("%Y-%m-%d")
        except Exception as e:
            # Log warnings internally but don't stop
            print(f"Earnings warning: {e}")
            pass

        # 3. FETCH OPTION DATES
        try:
            # This is where "No options found" usually happens
            opts = stock.options 
            if not opts:
                # Try one more time with a small sleep, might be a fluke
                time.sleep(1)
                opts = stock.options
                if not opts:
                    return {"error": f"YFinance returned no option dates. (Likely Rate Limit or No Data)"}
        except Exception as e:
            return {"error": f"Failed to fetch option dates: {str(e)}"}

        try:
            exp_dates = filter_dates(list(opts))
        except ValueError as ve:
            return {"error": str(ve)}

        # 4. FETCH CHAINS LOOP
        options_chains = {}
        total = len(exp_dates)
        
        for idx, exp_date in enumerate(exp_dates):
            # RATE LIMIT PROTECTION
            # Random sleep between 0.25s and 0.5s to act human
            time.sleep(random.uniform(0.25, 0.50))
            
            try:
                chain = get_options_with_retry(stock, exp_date)
                options_chains[exp_date] = chain
            except Exception as e:
                # Log specific error for this date
                print(f"Failed {exp_date}: {e}")
                continue

            if progress_callback:
                progress_callback((idx + 1) / total)

        if not options_chains:
            return {"error": "All option chain downloads failed."}

        # 5. FETCH PRICE HISTORY
        try:
            # 3 months for volatility calc
            hist = stock.history(period="3mo")
            if hist.empty:
                return {"error": "Price history is empty."}
            
            underlying_price = hist['Close'].iloc[-1]
        except Exception as e:
            return {"error": f"Failed to fetch price history: {str(e)}"}

        # 6. CALCULATIONS
        atm_iv = {}
        straddle = None 
        i_count = 0
        
        for exp_date, chain in options_chains.items():
            calls = chain.calls
            puts = chain.puts
            
            if calls.empty or puts.empty: continue

            # Find ATM IV
            call_diffs = (calls['strike'] - underlying_price).abs()
            call_idx = call_diffs.idxmin()
            call_iv = calls.loc[call_idx, 'impliedVolatility']

            put_diffs = (puts['strike'] - underlying_price).abs()
            put_idx = put_diffs.idxmin()
            put_iv = puts.loc[put_idx, 'impliedVolatility']

            # Average IV
            atm_iv_val = (call_iv + put_iv) / 2.0
            # Filter bad data (sometimes Yahoo sends 0.0 for IV)
            if atm_iv_val > 0.01:
                atm_iv[exp_date] = atm_iv_val

            # Straddle Cost (Nearest Expiry)
            if i_count == 0:
                c_mid = (calls.loc[call_idx, 'bid'] + calls.loc[call_idx, 'ask']) / 2
                p_mid = (puts.loc[put_idx, 'bid'] + puts.loc[put_idx, 'ask']) / 2
                straddle = c_mid + p_mid
            
            i_count += 1
        
        if not atm_iv:
            return {"error": "Could not determine valid ATM IVs (Data might be zero)."}
        
        # Spline Calc
        today = datetime.today().date()
        dtes = []
        ivs = []
        for exp_date, iv in atm_iv.items():
            edobj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            days = (edobj - today).days
            if days <= 0: days = 0.01
            dtes.append(days)
            ivs.append(iv)
            
        term_spline = build_term_structure(dtes, ivs)
        
        # Metrics
        try:
            ts_slope = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])
            iv30 = term_spline(30)
            rv30 = yang_zhang(hist)
            ratio = iv30 / rv30 if rv30 > 0 else 0
            avg_vol = hist['Volume'].rolling(30).mean().iloc[-1]
            exp_move = (straddle / underlying_price * 100) if straddle else 0
        except Exception as e:
            return {"error": f"Math error during calc: {str(e)}"}

        # Recommendation Logic
        pass_vol = avg_vol >= 1500000
        pass_ratio = ratio >= 1.25
        pass_slope = ts_slope <= -0.00406

        if pass_vol and pass_ratio and pass_slope:
            status = "RECOMMENDED"
        elif pass_slope and ((pass_vol and not pass_ratio) or (pass_ratio and not pass_vol)):
            status = "CONSIDER"
        else:
            status = "AVOID"

        return {
            "Symbol": ticker_symbol,
            "Status": status,
            "Next Earnings": next_earnings,
            "Avg Vol": avg_vol,
            "IV30/RV30": ratio,
            "TS Slope": ts_slope,
            "Exp Move %": exp_move,
            "pass_vol": pass_vol,
            "pass_ratio": pass_ratio,
            "pass_slope": pass_slope
        }

    except Exception as e:
        # This catches ANY crash and returns it as a string
        return {"error": f"CRITICAL: {str(e)}"}

# ==========================================
# GUI LOGIC
# ==========================================

def main():
    st.set_page_config(page_title="Earnings Checker", layout="wide")
    st.title("Earnings Position Checker (Debug Mode)")

    tab1, tab2 = st.tabs(["Single Ticker", "Batch / S&P 500"])

    # --- TAB 1 ---
    with tab1:
        with st.form("single"):
            ticker = st.text_input("Symbol", "AAPL")
            submitted = st.form_submit_button("Analyze")
        
        if submitted and ticker:
            with st.spinner(f"Scanning {ticker}..."):
                # Progress bar for single mode
                pbar = st.progress(0, text="Init...")
                def update_p(x): pbar.progress(x, text="Fetching Options...")
                
                res = compute_recommendation(ticker, update_p)
                pbar.empty()
            
            if "error" in res:
                st.error(f"❌ ERROR for {ticker}:\n\n{res['error']}")
            else:
                # Success UI
                color = "#28a745" if res['Status'] == "RECOMMENDED" else "#ffc107" if res['Status'] == "CONSIDER" else "#dc3545"
                st.markdown(f"<h2 style='color:{color}; border:1px solid {color}; padding:10px'>{res['Status']}</h2>", unsafe_allow_html=True)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Earnings", res['Next Earnings'])
                c2.metric("Avg Vol", f"{res['Avg Vol']/1e6:.2f}M", delta="PASS" if res['pass_vol'] else "FAIL")
                c3.metric("IV/RV Ratio", f"{res['IV30/RV30']:.2f}", delta="PASS" if res['pass_ratio'] else "FAIL")
                c4.metric("Slope", f"{res['TS Slope']:.5f}", delta="PASS" if res['pass_slope'] else "FAIL")
                st.info(f"Expected Move: {res['Exp Move %']:.2f}%")

    # --- TAB 2 ---
    with tab2:
        col_in, col_btn = st.columns([3,1])
        with col_in:
            batch_txt = st.text_area("Tickers (comma sep) or leave empty for S&P500", height=100)
        with col_btn:
            st.write("")
            st.write("")
            run_batch = st.button("Run Batch")
            stop_batch = st.button("STOP")

        if "stop" not in st.session_state: st.session_state.stop = False
        if stop_batch: st.session_state.stop = True

        if run_batch:
            st.session_state.stop = False
            if batch_txt.strip():
                tickers = [x.strip() for x in batch_txt.split(',') if x.strip()]
            else:
                with st.spinner("Fetching S&P 500..."):
                    tickers = get_sp500_tickers()
            
            container = st.empty()
            prog = st.progress(0, "Batch Starting...")
            results = []

            for i, t in enumerate(tickers):
                if st.session_state.stop:
                    st.warning("Stopped.")
                    break
                
                prog.progress((i)/len(tickers), f"Processing {t}...")
                
                # Call analysis
                data = compute_recommendation(t)
                
                if "error" in data:
                    # LOG THE ERROR IN THE TABLE
                    results.append({
                        "Ticker": t,
                        "Status": "ERROR", 
                        "Note": data['error'], # <--- SHOWS THE ERROR REASON
                        "Slope": 0, "Ratio": 0, "Vol": 0
                    })
                else:
                    results.append({
                        "Ticker": t,
                        "Status": data['Status'],
                        "Note": data['Next Earnings'],
                        "Slope": round(data['TS Slope'], 5),
                        "Ratio": round(data['IV30/RV30'], 2),
                        "Vol": round(data['Avg Vol']/1e6, 2)
                    })
                
                # Show partial dataframe
                df = pd.DataFrame(results)
                
                def color_row(row):
                    s = row['Status']
                    if s == 'RECOMMENDED': return ['background-color: #d4edda']*len(row)
                    if s == 'ERROR': return ['background-color: #f8d7da']*len(row)
                    if s == 'CONSIDER': return ['background-color: #fff3cd']*len(row)
                    return ['']*len(row)

                try:
                    container.dataframe(df.style.apply(color_row, axis=1), use_container_width=True)
                except:
                    container.dataframe(df, use_container_width=True)
                
                # Crucial sleep to prevent back-to-back 429
                time.sleep(1.0) 

            prog.empty()
            st.success("Done.")

if __name__ == "__main__":
    main()