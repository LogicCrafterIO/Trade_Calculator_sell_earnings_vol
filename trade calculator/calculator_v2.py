"""
DISCLAIMER: 
This software is provided solely for educational and research purposes. 
It is not intended to provide investment advice, and no investment recommendations are made herein. 
The developers are not financial advisors and accept no responsibility for any financial decisions or losses 
resulting from the use of this software. 
Always consult a professional financial advisor before making any investment decisions.
"""

import sys
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal


# -----------------------------
# ORIGINAL LOGIC (unchanged)
# -----------------------------
def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]  
            break
    
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")

def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = np.log(price_data['High'] / price_data['Open'])
    log_lo = np.log(price_data['Low'] / price_data['Open'])
    log_co = np.log(price_data['Close'] / price_data['Open'])
    
    log_oc = np.log(price_data['Open'] / price_data['Close'].shift(1))
    log_cc = np.log(price_data['Close'] / price_data['Close'].shift(1))

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc.pow(2).rolling(window).sum() / (window - 1)
    open_vol = log_oc.pow(2).rolling(window).sum() / (window - 1)
    window_rs = rs.rolling(window).sum() / (window - 1)

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).pow(0.5) * np.sqrt(trading_periods)

    return result.iloc[-1]

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:  
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:
            return float(spline(dte))

    return term_spline

def get_current_price(ticker):
    todays_data = ticker.history(period="1d")
    return todays_data["Close"][0]

def compute_recommendation(ticker):
    ticker = ticker.strip().upper()
    if not ticker:
        return {"error": "No stock symbol provided."}

    stock = yf.Ticker(ticker)
    if len(stock.options) == 0:
        return {"error": f"No options found for '{ticker}'."}

    try:
        exp_dates = filter_dates(stock.options)
    except:
        return {"error": "Not enough option data."}

    options_chains = {d: stock.option_chain(d) for d in exp_dates}

    try:
        underlying_price = get_current_price(stock)
    except:
        return {"error": "Unable to retrieve stock price."}

    atm_iv = {}
    straddle = None
    i = 0

    for exp, chain in options_chains.items():
        calls = chain.calls
        puts = chain.puts
        if calls.empty or puts.empty:
            continue

        call_idx = (calls["strike"] - underlying_price).abs().idxmin()
        put_idx = (puts["strike"] - underlying_price).abs().idxmin()

        call_iv = calls.loc[call_idx, "impliedVolatility"]
        put_iv  = puts.loc[put_idx, "impliedVolatility"]

        atm_iv[exp] = (call_iv + put_iv) / 2

        if i == 0:
            call_mid = (calls.loc[call_idx, "bid"] + calls.loc[call_idx, "ask"]) / 2
            put_mid  = (puts.loc[put_idx, "bid"] + puts.loc[put_idx, "ask"]) / 2
            straddle = call_mid + put_mid
        i += 1

    today = datetime.today().date()
    dtes = [(datetime.strptime(e, "%Y-%m-%d").date() - today).days for e in atm_iv]
    ivs = list(atm_iv.values())

    term_spline = build_term_structure(dtes, ivs)

    price_history = stock.history(period="3mo")
    iv30_rv30 = term_spline(30) / yang_zhang(price_history)

    avg_volume = price_history["Volume"].rolling(30).mean().iloc[-1]
    expected_move = f"{round(straddle / underlying_price * 100, 2)}%"

    return {
        "avg_volume": avg_volume >= 1500000,
        "iv30_rv30": iv30_rv30 >= 1.25,
        "ts_slope_0_45": (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0]) <= -0.00406,
        "expected_move": expected_move
    }


# -----------------------------
# THREAD WORKER
# -----------------------------
class Worker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker

    def run(self):
        result = compute_recommendation(self.ticker)
        self.finished.emit(result)


# -----------------------------
# GUI (PyQt6)
# -----------------------------
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Earnings Position Checker")

        self.label = QLabel("Enter Stock Symbol:")
        self.input = QLineEdit()
        self.button = QPushButton("Submit")
        self.result = QLabel("")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.input)
        layout.addWidget(self.button)
        layout.addWidget(self.result)

        self.setLayout(layout)

        self.button.clicked.connect(self.start_compute)

    def start_compute(self):
        ticker = self.input.text()
        if not ticker:
            QMessageBox.warning(self, "Error", "Please enter a stock symbol.")
            return

        self.result.setText("Loading…")

        self.thread = Worker(ticker)
        self.thread.finished.connect(self.show_result)
        self.thread.start()

    def show_result(self, result):
        if "error" in result:
            self.result.setText(result["error"])
            return

        avg_ok = result["avg_volume"]
        iv_ok = result["iv30_rv30"]
        ts_ok = result["ts_slope_0_45"]
        move = result["expected_move"]

        if avg_ok and iv_ok and ts_ok:
            status = "Recommended"
        elif ts_ok and (avg_ok or iv_ok):
            status = "Consider"
        else:
            status = "Avoid"

        text = f"""
<b>{status}</b><br>
avg_volume: {"PASS" if avg_ok else "FAIL"}<br>
iv30_rv30: {"PASS" if iv_ok else "FAIL"}<br>
ts_slope_0_45: {"PASS" if ts_ok else "FAIL"}<br>
Expected Move: {move}
"""
        self.result.setText(text)


# -----------------------------
# MAIN
# -----------------------------
def gui():
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    gui()
