import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import time
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

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    # Parse and sort
    dt_objs = []
    for d in dates:
        try:
            dt_objs.append(datetime.strptime(d, "%Y-%m-%d").date())
        except:
            continue 
            
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
        return 0 
        
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
# CORE LOGIC
# ==========================================

def get_options_with_retry(stock, exp_date, retries=3):
    """Retries fetching option chain if it fails."""
    for i in range(retries):
        try:
            if i > 0: time.sleep(1 * (i+1)) 
            return stock.option_chain(exp_date)
        except Exception as e:
            if i == retries - 1: raise e 
            pass

def get_atm_details(chain, underlying_price):
    """Helper to extract ATM Call/Put Price (Mid) and IV from a chain."""
    try:
        calls = chain.calls
        puts = chain.puts
        
        if calls.empty or puts.empty:
            return None

        # Find ATM Call
        c_diffs = (calls['strike'] - underlying_price).abs()
        c_idx = c_diffs.idxmin()
        c_iv = calls.loc[c_idx, 'impliedVolatility']
        c_mid = (calls.loc[c_idx, 'bid'] + calls.loc[c_idx, 'ask']) / 2.0

        # Find ATM Put
        p_diffs = (puts['strike'] - underlying_price).abs()
        p_idx = p_diffs.idxmin()
        p_iv = puts.loc[p_idx, 'impliedVolatility']
        p_mid = (puts.loc[p_idx, 'bid'] + puts.loc[p_idx, 'ask']) / 2.0

        return {
            'c_price': c_mid, 'c_iv': c_iv,
            'p_price': p_mid, 'p_iv': p_iv
        }
    except:
        return None

def compute_recommendation(ticker_symbol, progress_callback=None):
    try:
        ticker_symbol = ticker_symbol.strip().upper()
        stock = yf.Ticker(ticker_symbol)
        
        # 1. FETCH EARNINGS
        next_earnings = "N/A"
        try:
            cal = stock.calendar
            if cal and isinstance(cal, dict) and 'Earnings Date' in cal:
                dates = cal['Earnings Date']
                now = datetime.now().date()
                future_dates = [d for d in dates if d >= now]
                if future_dates:
                    next_earnings = future_dates[0].strftime("%Y-%m-%d")
        except:
            pass

        # 2. FETCH OPTION DATES
        try:
            opts = stock.options 
            if not opts:
                time.sleep(1)
                opts = stock.options
                if not opts:
                    return {"error": f"YFinance returned no option dates."}
        except Exception as e:
            return {"error": f"Failed to fetch option dates: {str(e)}"}

        try:
            exp_dates = filter_dates(list(opts))
        except ValueError as ve:
            return {"error": str(ve)}

        # 3. FETCH CHAINS LOOP
        options_chains = {}
        total = len(exp_dates)
        
        for idx, exp_date in enumerate(exp_dates):
            # Sleep to avoid 429
            time.sleep(random.uniform(0.25, 0.50))
            
            try:
                chain = get_options_with_retry(stock, exp_date)
                options_chains[exp_date] = chain
            except Exception as e:
                continue

            if progress_callback:
                progress_callback((idx + 1) / total)

        if not options_chains:
            return {"error": "All option chain downloads failed."}

        # 4. FETCH PRICE HISTORY
        try:
            hist = stock.history(period="3mo")
            if hist.empty:
                return {"error": "Price history is empty."}
            underlying_price = hist['Close'].iloc[-1]
        except Exception as e:
            return {"error": f"Failed to fetch price history: {str(e)}"}

        # 5. CALCULATIONS
        atm_iv = {}
        straddle = None 
        i_count = 0
        
        for exp_date, chain in options_chains.items():
            calls = chain.calls
            puts = chain.puts
            if calls.empty or puts.empty: continue

            call_diffs = (calls['strike'] - underlying_price).abs()
            call_idx = call_diffs.idxmin()
            call_iv = calls.loc[call_idx, 'impliedVolatility']

            put_diffs = (puts['strike'] - underlying_price).abs()
            put_idx = put_diffs.idxmin()
            put_iv = puts.loc[put_idx, 'impliedVolatility']

            atm_iv_val = (call_iv + put_iv) / 2.0
            if atm_iv_val > 0.01:
                atm_iv[exp_date] = atm_iv_val

            # Standard Straddle Calc (Nearest)
            if i_count == 0:
                c_mid = (calls.loc[call_idx, 'bid'] + calls.loc[call_idx, 'ask']) / 2
                p_mid = (puts.loc[put_idx, 'bid'] + puts.loc[put_idx, 'ask']) / 2
                straddle = c_mid + p_mid
            
            i_count += 1
        
        if not atm_iv:
            return {"error": "Could not determine valid ATM IVs."}
        
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
        
        try:
            ts_slope = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])
            iv30 = term_spline(30)
            rv30 = yang_zhang(hist)
            ratio = iv30 / rv30 if rv30 > 0 else 0
            avg_vol = hist['Volume'].rolling(30).mean().iloc[-1]
            exp_move = (straddle / underlying_price * 100) if straddle else 0
        except Exception as e:
            return {"error": f"Math error: {str(e)}"}

        pass_vol = avg_vol >= 1500000
        pass_ratio = ratio >= 1.25
        pass_slope = ts_slope <= -0.00406

        if pass_vol and pass_ratio and pass_slope:
            status = "RECOMMENDED"
        elif pass_slope and ((pass_vol and not pass_ratio) or (pass_ratio and not pass_vol)):
            status = "CONSIDER"
        else:
            status = "AVOID"

        # ==========================================
        # CALENDAR SPREAD DATA EXTRACTION
        # ==========================================
        cal_data = {
            "Near Call Px": 0, "Near Call IV": 0,
            "30d Call Px": 0, "30d Call IV": 0,
            "Call Spread Debit": 0,
            "Near Put Px": 0, "Near Put IV": 0,
            "30d Put Px": 0, "30d Put IV": 0,
            "Put Spread Debit": 0
        }

        if status in ["RECOMMENDED", "CONSIDER"]:
            try:
                # 1. Identify Dates
                # Nearest is index 0
                near_date = exp_dates[0]
                
                # 30 Day is date closest to Today + 30
                target_date = today + timedelta(days=30)
                # exp_dates are strings, convert to date objs to compare
                date_objs = [datetime.strptime(d, "%Y-%m-%d").date() for d in exp_dates]
                
                # Find index of closest date
                closest_idx = min(range(len(date_objs)), key=lambda i: abs((date_objs[i] - target_date).days))
                far_date = exp_dates[closest_idx]

                # 2. Extract Data
                near_details = get_atm_details(options_chains[near_date], underlying_price)
                far_details = get_atm_details(options_chains[far_date], underlying_price)

                if near_details and far_details:
                    # Call Side
                    cal_data["Near Call Px"] = near_details['c_price']
                    cal_data["Near Call IV"] = near_details['c_iv']
                    cal_data["30d Call Px"] = far_details['c_price']
                    cal_data["30d Call IV"] = far_details['c_iv']
                    # Calendar Debit = Long (Far) - Short (Near)
                    cal_data["Call Spread Debit"] = far_details['c_price'] - near_details['c_price']

                    # Put Side
                    cal_data["Near Put Px"] = near_details['p_price']
                    cal_data["Near Put IV"] = near_details['p_iv']
                    cal_data["30d Put Px"] = far_details['p_price']
                    cal_data["30d Put IV"] = far_details['p_iv']
                    cal_data["Put Spread Debit"] = far_details['p_price'] - near_details['p_price']
            except Exception as e:
                print(f"Error extracting calendar data: {e}")

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
            "pass_slope": pass_slope,
            **cal_data # Merge calendar data
        }

    except Exception as e:
        return {"error": f"CRITICAL: {str(e)}"}

# ==========================================
# GUI LOGIC
# ==========================================

def main():
    st.set_page_config(page_title="Earnings Checker", layout="wide")
    st.title("Earnings Position Checker")

    tab1, tab2 = st.tabs(["Single Ticker", "Batch Analysis"])

    # --- TAB 1: SINGLE TICKER ---
    with tab1:
        with st.form("single"):
            ticker = st.text_input("Symbol", "AAPL")
            submitted = st.form_submit_button("Analyze")
        
        if submitted and ticker:
            with st.spinner(f"Scanning {ticker}..."):
                pbar = st.progress(0, text="Init...")
                def update_p(x): pbar.progress(x, text="Fetching Options...")
                res = compute_recommendation(ticker, update_p)
                pbar.empty()
            
            if "error" in res:
                st.error(f"❌ ERROR for {ticker}: {res['error']}")
            else:
                if res['Status'] == "RECOMMENDED":
                    bg_color = "#d4edda"
                    text_color = "#155724"
                elif res['Status'] == "CONSIDER":
                    bg_color = "#fff3cd"
                    text_color = "#856404"
                else:
                    bg_color = "#f8d7da"
                    text_color = "#721c24"
                
                st.markdown(f"<h2 style='background-color:{bg_color}; color:{text_color}; border:2px solid {text_color}; padding:12px; border-radius:6px; text-align:center; font-weight:bold'>{res['Status']}</h2>", unsafe_allow_html=True)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Next Earnings", res['Next Earnings'])
                c2.metric("Avg Vol (>1.5M)", f"{res['Avg Vol']/1e6:.2f}M", delta="PASS" if res['pass_vol'] else "FAIL")
                c3.metric("Ratio (>1.25)", f"{res['IV30/RV30']:.2f}", delta="PASS" if res['pass_ratio'] else "FAIL")
                c4.metric("Slope (<-0.004)", f"{res['TS Slope']:.5f}", delta="PASS" if res['pass_slope'] else "FAIL")
                st.info(f"Expected Move: {res['Exp Move %']:.2f}%")

                # SHOW CALENDAR DATA IF RELEVANT
                if res['Status'] in ["RECOMMENDED", "CONSIDER"]:
                    st.divider()
                    st.subheader("Long Calendar Spread Analysis (ATM)")
                    
                    col_call, col_put = st.columns(2)
                    
                    with col_call:
                        st.markdown("### 📞 CALL SIDE")
                        st.write(f"**Near Call:** ${res['Near Call Px']:.2f} (IV: {res['Near Call IV']:.2%})")
                        st.write(f"**30d Call:** ${res['30d Call Px']:.2f} (IV: {res['30d Call IV']:.2%})")
                        st.metric("Net Debit (Call)", f"${res['Call Spread Debit']:.2f}")

                    with col_put:
                        st.markdown("### 📉 PUT SIDE")
                        st.write(f"**Near Put:** ${res['Near Put Px']:.2f} (IV: {res['Near Put IV']:.2%})")
                        st.write(f"**30d Put:** ${res['30d Put Px']:.2f} (IV: {res['30d Put IV']:.2%})")
                        st.metric("Net Debit (Put)", f"${res['Put Spread Debit']:.2f}")

    # --- TAB 2: BATCH ---
    with tab2:
        st.markdown("### 1. Upload Tickers (Optional)")
        uploaded_file = st.file_uploader("Upload CSV (Column header must be 'tickers')", type=['csv'])
        
        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if st.session_state.get('last_processed_file_id') != file_id:
                try:
                    df_up = pd.read_csv(uploaded_file)
                    cols = [c.lower() for c in df_up.columns]
                    if 'tickers' in cols:
                        idx = cols.index('tickers')
                        raw_tickers = df_up.iloc[:, idx].dropna().astype(str).tolist()
                        cleaned_tickers = [t.strip().upper() for t in raw_tickers if t.strip()]
                        combined_str = ", ".join(cleaned_tickers)
                        
                        st.session_state['batch_tickers_input'] = combined_str
                        st.session_state['last_processed_file_id'] = file_id
                        
                        st.success(f"Loaded {len(cleaned_tickers)} tickers from CSV.")
                    else:
                        st.error("CSV must have a column named 'tickers'.")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        st.markdown("### 2. Review & Run")
        
        if 'batch_tickers_input' not in st.session_state:
            st.session_state['batch_tickers_input'] = ""

        batch_txt = st.text_area(
            "Enter tickers (comma separated)", 
            height=100, 
            key="batch_tickers_input" 
        )

        col_btn1, col_btn2 = st.columns([1,5])
        with col_btn1:
            run_batch = st.button("Run Batch")
        with col_btn2:
            stop_batch = st.button("STOP")

        if "stop" not in st.session_state: st.session_state.stop = False
        if stop_batch: st.session_state.stop = True

        if run_batch:
            st.session_state.stop = False
            if not batch_txt.strip():
                st.warning("Please enter tickers or upload a CSV.")
            else:
                tickers = [x.strip() for x in batch_txt.split(',') if x.strip()]
                
                container = st.empty()
                prog = st.progress(0, "Starting...")
                results = []

                for i, t in enumerate(tickers):
                    if st.session_state.stop:
                        st.warning("Stopped by user.")
                        break
                    
                    prog.progress((i)/len(tickers), f"Processing {t} ({i+1}/{len(tickers)})...")
                    
                    data = compute_recommendation(t)
                    
                    row = {}
                    if "error" in data:
                        row = {
                            "Ticker": t,
                            "Status": "ERROR", 
                            "Note": data['error'],
                            # Fill empty for structure
                            "Slope": 0, "Ratio": 0, "Vol": 0, "ExpMove%": 0,
                            "Call Debit": 0, "Put Debit": 0,
                            "Near Call IV": 0, "30d Call IV": 0,
                            "Near Put IV": 0, "30d Put IV": 0
                        }
                    else:
                        row = {
                            "Ticker": t,
                            "Status": data['Status'],
                            "Note": data['Next Earnings'],
                            "Slope": round(data['TS Slope'], 5),
                            "Ratio": round(data['IV30/RV30'], 2),
                            "Vol": f"{data['Avg Vol']/1e6:.2f}M",
                            "ExpMove%": round(data['Exp Move %'], 2)
                        }
                        
                        # Add Calendar Data if Status is good
                        if data['Status'] in ["RECOMMENDED", "CONSIDER"]:
                            row["Call Debit"] = round(data["Call Spread Debit"], 2)
                            row["Put Debit"] = round(data["Put Spread Debit"], 2)
                            row["Near Call IV"] = round(data["Near Call IV"], 3)
                            row["30d Call IV"] = round(data["30d Call IV"], 3)
                            row["Near Put IV"] = round(data["Near Put IV"], 3)
                            row["30d Put IV"] = round(data["30d Put IV"], 3)
                            
                            # Add prices if you want table to be super wide, 
                            # but let's keep it readable with just debits and IVs
                            # If you really want all 10 cols, uncomment below:
                            row["Near Call $"] = round(data["Near Call Px"], 2)
                            row["30d Call $"] = round(data["30d Call Px"], 2)
                            row["Near Put $"] = round(data["Near Put Px"], 2)
                            row["30d Put $"] = round(data["30d Put Px"], 2)

                    results.append(row)
                    
                    df = pd.DataFrame(results)
                    
                    def color_row(row):
                        s = row['Status']
                        style = []
                        if s == 'RECOMMENDED': 
                            return ['background-color: #d4edda; color: #155724; font-weight: bold']*len(row)
                        if s == 'ERROR': 
                            return ['background-color: #f8d7da; color: #721c24; font-weight: bold']*len(row)
                        if s == 'CONSIDER': 
                            return ['background-color: #fff3cd; color: #856404; font-weight: bold']*len(row)
                        return ['']*len(row)

                    try:
                        container.dataframe(df.style.apply(color_row, axis=1), use_container_width=True)
                    except:
                        container.dataframe(df, use_container_width=True)
                    
                    time.sleep(1.0) 

                prog.empty()
                st.success("Batch Complete.")

if __name__ == "__main__":
    main()