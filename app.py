"""
Investment Banking Valuation Workbench Pro v2.1
===============================================
A professional-grade DCF and comparable companies analysis tool for equity valuation.

Author: Professional Development Team
Last Updated: 2024
Dependencies: See requirements.txt
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import statsmodels.api as sm
import io
import math
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime
import warnings as py_warnings

py_warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="IB Valuation Workbench Pro",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-card { background-color: #121417; border: 1px solid #2e2e2e; padding: 12px; border-radius: 6px; text-align: center; }
    .hero-val { font-size: 2.1rem; font-weight: 700; color: #4caf50; }
    .hero-label { font-size: 0.85rem; color: #b0bec5; text-transform: uppercase; }
    .warning-box { background-color: #3e2723; border-left: 5px solid #ff6f00; padding: 10px; margin-bottom: 8px; font-size: 0.9rem; }
    .error-box { background-color: #b71c1c; border-left: 5px solid #f44336; padding: 10px; margin-bottom: 8px; font-size: 0.9rem; color: #fff; }
    .info-box { background-color: #01579b; border-left: 5px solid #03a9f4; padding: 10px; margin-bottom: 8px; font-size: 0.9rem; }
    .success-box { background-color: #1b5e20; border-left: 5px solid #4caf50; padding: 10px; margin-bottom: 8px; font-size: 0.9rem; }
    .dataframe { font-size: 0.8rem !important; }
    </style>
""", unsafe_allow_html=True)

# Constants
SECTOR_MAP = {
    "RELIANCE": ["ONGC.NS", "BPCL.NS", "IOC.NS", "OIL.NS"],
    "TCS": ["INFY.NS", "HCLTECH.NS", "WIPRO.NS", "LTIM.NS"],
    "INFY": ["TCS.NS", "HCLTECH.NS", "TECHM.NS", "WIPRO.NS"],
    "HDFCBANK": ["ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "ITC": ["HUL.NS", "NESTLEIND.NS", "BRITANNIA.NS", "GODREJCP.NS"],
    "TATAMOTORS": ["MARUTI.NS", "M&M.NS", "ASHOKLEY.NS", "EICHERMOT.NS"],
    "ZOMATO": ["NAUKRI.NS", "PAYTM.NS", "NYKAA.NS", "DELHIVERY.NS"],
    "SWIGGY": ["ZOMATO.NS", "NAUKRI.NS", "PAYTM.NS"]
}

MAX_TERMINAL_GROWTH_INDIA = 0.07
MIN_WACC = 0.05
MAX_WACC = 0.30
MAX_PEERS = 15
WACC_LOW_THRESHOLD = 0.08
WACC_HIGH_THRESHOLD = 0.15

# ==================== DATA QUALITY TRACKER ====================

class DataQualityTracker:
    """Centralized tracking of data quality issues, assumptions, and errors."""
    
    def __init__(self):
        self.issues: List[Tuple[str, str]] = []
        self.critical_errors: List[Tuple[str, str]] = []
        self.assumptions: List[Tuple[str, str]] = []
        self.successes: List[Tuple[str, str]] = []

    def add_warning(self, msg: str):
        """Add a non-critical data quality warning."""
        self.issues.append(("⚠️", msg))

    def add_error(self, msg: str):
        """Add a critical error that should stop execution."""
        self.critical_errors.append(("❌", msg))

    def add_assumption(self, msg: str):
        """Document an assumption made in the analysis."""
        self.assumptions.append(("📊", msg))
    
    def add_success(self, msg: str):
        """Document successful data retrieval."""
        self.successes.append(("✅", msg))

    def has_critical_errors(self) -> bool:
        """Check if any critical errors exist."""
        return len(self.critical_errors) > 0

    def render(self):
        """Render all tracked issues in Streamlit UI."""
        if self.critical_errors:
            st.error("**🚨 Critical Data Issues - Review Required**")
            for icon, msg in self.critical_errors:
                st.markdown(f"<div class='error-box'>{icon} {msg}</div>", unsafe_allow_html=True)

        if self.assumptions:
            with st.expander("📊 Key Assumptions & Data Sources", expanded=True):
                for icon, msg in self.assumptions:
                    st.markdown(f"<div class='info-box'>{icon} {msg}</div>", unsafe_allow_html=True)

        if self.issues:
            with st.expander(f"⚠️ Data Quality Warnings ({len(self.issues)})", expanded=False):
                for icon, msg in self.issues:
                    st.markdown(f"<div class='warning-box'>{icon} {msg}</div>", unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default return value."""
    try:
        if b == 0 or b is None or math.isnan(b):
            return default
        return a / b
    except Exception:
        return default

def clean_number(v: Any) -> float:
    """Convert NaN/Inf to 0.0 for safe Excel export."""
    if isinstance(v, (int, float, np.number)):
        if math.isnan(v) or math.isinf(v):
            return 0.0
    return v

@st.cache_data(ttl=3600, show_spinner=False)
def get_risk_free_rate() -> float:
    """Fetch current India 10Y Government Bond yield."""
    try:
        url = "https://www.marketwatch.com/investing/bond/tmbmkin-10y?countrycode=bx"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(r.text, "html.parser")
        tag = soup.find("bg-quote", class_="value")
        if tag:
            return float(tag.text.strip().replace("%", "").replace(",", "")) / 100.0
    except Exception:
        pass
    return 0.071

def clean_peer_list(raw_text: str) -> List[str]:
    """Parse comma-separated ticker list."""
    if not raw_text:
        return []
    clean = raw_text.replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace("\n", ",")
    return [x.strip().upper() for x in clean.split(",") if x.strip()]

# ==================== FINANCIAL DATA ENGINE ====================

@st.cache_resource(show_spinner=False, ttl=900)
def load_ticker_payload(ticker: str) -> Tuple[Optional[Dict[str, Any]], DataQualityTracker]:
    """
    Load comprehensive financial data for a ticker.
    
    Returns:
        Tuple of (data_dict, DataQualityTracker) or (None, tracker) on critical failure
    """
    dq = DataQualityTracker()
    
    try:
        t = yf.Ticker(ticker)
    except Exception as e:
        dq.add_error(f"Cannot initialize ticker '{ticker}': {e}")
        return None, dq

    # ========== Market Data ==========
    price, shares, mcap = 0.0, 0, 0

    try:
        fast = getattr(t, "fast_info", {})
        price = float(fast.get("last_price", 0))
        shares = int(fast.get("shares", 0))
        mcap = price * shares
    except Exception:
        pass

    if price == 0:
        try:
            info = t.info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
            if shares == 0:
                shares = info.get("sharesOutstanding") or 0
            if mcap == 0:
                mcap = info.get("marketCap") or 0
        except Exception:
            pass

    if price == 0:
        dq.add_error("Could not fetch current price. Ticker may be delisted or data unavailable.")
        return None, dq

    if shares == 0:
        dq.add_error("Shares outstanding is zero or unavailable. Cannot calculate per-share metrics.")
        return None, dq

    dq.add_assumption(f"Current Price: ₹{price:,.2f} | Shares Outstanding: {shares/1e6:,.1f}M")

    # ========== Financials ==========
    inc = getattr(t, "quarterly_income_stmt", None)
    bal = getattr(t, "quarterly_balance_sheet", None)
    cf = getattr(t, "quarterly_cashflow", None)

    use_annual = False
    if inc is None or inc.empty:
        inc = getattr(t, "income_stmt", None)
        bal = getattr(t, "balance_sheet", None)
        cf = getattr(t, "cashflow", None)
        use_annual = True
        dq.add_warning("Quarterly data unavailable. Using annual financials (may be stale).")
        dq.add_assumption("Financial Period: Annual (TTM approximation)")
    else:
        dq.add_assumption("Financial Period: Quarterly (Last 4 quarters)")
        dq.add_success("Successfully retrieved quarterly financial statements")

    def get_val(df, keys):
        if df is None or df.empty:
            return 0.0
        for k in keys:
            if k in df.index:
                if use_annual:
                    return float(df.loc[k].iloc[0] or 0)
                try:
                    return float(df.loc[k].iloc[:4].sum() or 0)
                except Exception:
                    return 0.0
        return 0.0

    revenue = get_val(inc, ["Total Revenue", "Revenue"])
    interest = get_val(inc, ["Interest Expense", "Interest Expense Non Operating"])
    tax_exp = get_val(inc, ["Tax Provision", "Income Tax Expense"])
    net_income = get_val(inc, ["Net Income", "Net Income Common Stockholders"])

    if revenue == 0:
        dq.add_error("Revenue is zero or missing. Cannot proceed with valuation.")
        return None, dq

    da = get_val(cf, ["Depreciation And Amortization", "Depreciation"])
    ebitda = get_val(inc, ["EBITDA", "Normalized EBITDA"])

    if ebitda == 0:
        ebitda = net_income + tax_exp + interest + da
        dq.add_warning("EBITDA not reported. Derived as Net Income + Tax + Interest + D&A.")
        dq.add_assumption(f"EBITDA (Derived): ₹{ebitda/1e7:,.0f} Cr")
    else:
        dq.add_assumption(f"EBITDA (Reported): ₹{ebitda/1e7:,.0f} Cr")

    if ebitda <= 0:
        dq.add_error(f"EBITDA is negative or zero (₹{ebitda/1e7:,.0f} Cr). Company may be loss-making.")

    ebit = ebitda - da
    if ebit <= 0:
        ebit = net_income + tax_exp + interest
        if ebit <= 0:
            dq.add_warning(f"EBIT is negative (₹{ebit/1e7:,.0f} Cr). Impacts WACC and DCF.")

    ocf = get_val(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = abs(get_val(cf, ["Capital Expenditure", "Net PPE Purchase And Sale"]))

    def get_bs(df, keys):
        if df is None or df.empty:
            return 0.0
        for k in keys:
            if k in df.index:
                return float(df.loc[k].iloc[0] or 0)
        return 0.0

    def get_bs_prev(df, keys):
        if df is None or df.empty or len(df.columns) < 2:
            return 0.0
        for k in keys:
            if k in df.index:
                return float(df.loc[k].iloc[1] or 0)
        return 0.0

    curr_assets = get_bs(bal, ["Current Assets"])
    prev_assets = get_bs_prev(bal, ["Current Assets"])
    curr_liab = get_bs(bal, ["Current Liabilities"])
    prev_liab = get_bs_prev(bal, ["Current Liabilities"])

    wc_current = curr_assets - curr_liab
    wc_prev = prev_assets - prev_liab
    delta_wc = wc_current - wc_prev if wc_prev != 0 else 0

    # FCF Calculation
    fcf = ocf - capex

    if fcf == 0 or ocf == 0:
        tax_rate = safe_div(tax_exp, (net_income + tax_exp), 0.25)
        nopat = ebit * (1 - tax_rate)
        fcf = nopat + da - capex - delta_wc
        dq.add_warning("Operating Cash Flow unavailable. FCF calculated as NOPAT + D&A - CapEx - ΔWC.")
        dq.add_assumption(f"FCF (Calculated): ₹{fcf/1e7:,.0f} Cr")
    else:
        dq.add_assumption(f"FCF (OCF-CapEx): ₹{fcf/1e7:,.0f} Cr")

    if fcf <= 0:
        dq.add_error(f"Free Cash Flow is negative (₹{fcf/1e7:,.0f} Cr). DCF may not be appropriate.")

    # Balance Sheet
    total_debt = get_bs(bal, ["Total Debt", "Long Term Debt And Capital Lease Obligation"])
    cash = get_bs(bal, ["Cash And Cash Equivalents", "Cash"])
    total_assets = get_bs(bal, ["Total Assets"])
    total_equity = get_bs(bal, ["Total Equity Gross Minority Interest", "Stockholders Equity"])

    if total_debt == 0 and interest > 0:
        dq.add_warning("Total Debt is zero but interest exists. Balance sheet data may be incomplete.")

    capital_employed = total_assets - curr_liab if total_assets > 0 else total_equity + total_debt

    # Margins & Returns
    ebitda_margin = safe_div(ebitda, revenue) if revenue > 0 else 0
    fcf_margin = safe_div(fcf, revenue) if revenue > 0 else 0
    net_margin = safe_div(net_income, revenue) if revenue > 0 else 0

    if fcf_margin < 0:
        dq.add_warning(f"Negative FCF margin ({fcf_margin*100:.1f}%). High growth with negative FCF is risky.")
    elif fcf_margin > 0.3:
        dq.add_assumption(f"High FCF margin ({fcf_margin*100:.1f}%). Verify sustainability of cash generation.")

    roce = safe_div(ebit * 0.75, capital_employed) if capital_employed > 0 else 0

    net_debt = total_debt - cash
    if net_debt < 0:
        dq.add_assumption(f"Net Cash position: ₹{abs(net_debt)/1e7:,.0f} Cr (Cash > Debt)")

    return {
        "price": price, "shares": shares, "mcap": mcap,
        "revenue": revenue, "ebitda": ebitda, "ebit": ebit,
        "net_income": net_income, "interest": interest, "fcf": fcf,
        "tax_exp": tax_exp, "da": da, "debt": total_debt,
        "cash": cash, "net_debt": net_debt,
        "ev": mcap + net_debt,
        "ocf": ocf, "capex": capex, "delta_wc": delta_wc,
        "capital_employed": capital_employed,
        "ebitda_margin": ebitda_margin,
        "fcf_margin": fcf_margin,
        "net_margin": net_margin,
        "roce": roce,
        "wc_current": wc_current
    }, dq

# ==================== BETA CALCULATION ====================

@st.cache_data(ttl=900, show_spinner=False)
def calculate_beta_regression(
    ticker: str, 
    market_symbol: str = "^NSEI", 
    period: str = "2y",
    interval: str = "1d"
) -> Tuple[float, float, Optional[Any], str, List[str]]:
    """
    Calculate beta using linear regression.
    
    Args:
        ticker: Stock ticker
        market_symbol: Market index ticker
        period: Historical period (6mo, 1y, 2y, 5y)
        interval: Data frequency (1d, 1wk)
    
    Returns:
        Tuple of (beta, r_squared, figure, status_msg, issues_list)
    """
    issues: List[str] = []
    
    try:
        df = yf.download(
            [ticker, market_symbol], 
            period=period, 
            interval=interval,
            progress=False, 
            auto_adjust=True
        )["Close"]

        if df.empty:
            return 1.0, 0.0, None, "No price data", ["❌ No historical price data. Using default beta=1.0"]

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        returns = df.pct_change().dropna()

        if ticker not in returns.columns or market_symbol not in returns.columns:
            return 1.0, 0.0, None, "Missing columns", ["❌ Price data incomplete. Using default beta=1.0"]

        if len(returns) < 50:
            issues.append(f"⚠️ Only {len(returns)} observations. Beta estimate may be unreliable.")

        y = returns[ticker]
        x = returns[market_symbol]

        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = float(model.params.iloc[1])
        r2 = float(model.rsquared)

        if r2 < 0.3:
            issues.append(f"⚠️ Low R² ({r2:.2f}). Stock may not track market well.")

        if abs(beta) > 3:
            issues.append(f"⚠️ Extreme beta ({beta:.2f}). Verify or use manual input.")

        fig = px.scatter(
            x=x, y=y, trendline="ols", opacity=0.5,
            labels={"x": "Market Return", "y": "Stock Return"},
            title=f"Beta Regression: {beta:.3f} (R²: {r2:.3f}, n={len(returns)})"
        )
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,20,0.5)")

        return beta, r2, fig, "Success", issues

    except Exception as e:
        return 1.0, 0.0, None, str(e), [f"❌ Beta calculation failed: {e}. Using default beta=1.0"]

def calculate_unlevered_beta(levered_beta: float, debt: float, equity: float, tax_rate: float) -> float:
    """Calculate unlevered beta: βu = βl / [1 + (1-T)(D/E)]"""
    if equity == 0:
        return levered_beta
    de_ratio = debt / equity
    return levered_beta / (1 + (1 - tax_rate) * de_ratio)

# ==================== PEER ANALYSIS ====================

@st.cache_data(ttl=900, show_spinner=False)
def get_detailed_peers(ticker_list: List[str]) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Fetch detailed financial data for peer companies.
    
    Returns:
        Tuple of (DataFrame, issues_list)
    """
    rows = []
    issues: List[str] = []

    if len(ticker_list) > MAX_PEERS:
        issues.append(f"⚠️ Analyzing {len(ticker_list)} peers may be slow. Consider reducing to top {MAX_PEERS}.")

    failed = 0
    for t in ticker_list:
        try:
            clean_t = f"{t.replace('.NS', '')}.NS" if not t.endswith(".NS") else t
            tk = yf.Ticker(clean_t)
            i = tk.info

            mcap = i.get("marketCap", 0)
            if mcap == 0:
                failed += 1
                continue

            ev = i.get("enterpriseValue", 0)
            revenue = i.get("totalRevenue", 0)
            ebitda = i.get("ebitda", 0)

            if ev == 0 or revenue == 0:
                issues.append(f"⚠️ {clean_t.replace('.NS', '')}: Missing EV or Revenue data")

            rows.append({
                "Ticker": clean_t.replace(".NS", ""),
                "Price": i.get("currentPrice", 0),
                "Market Cap (Cr)": mcap / 1e7,
                "EV (Cr)": (ev if ev > 0 else mcap) / 1e7,
                "Revenue (Cr)": revenue / 1e7,
                "EBITDA (Cr)": ebitda / 1e7 if ebitda else 0,
                "EV/Revenue": safe_div(ev if ev > 0 else mcap, revenue),
                "EV/EBITDA": i.get("enterpriseToEbitda", 0) if ebitda else 0,
                "P/E": i.get("trailingPE", 0),
                "EBITDA Margin": safe_div(ebitda, revenue) if revenue > 0 else 0
            })
        except Exception as e:
            failed += 1
            issues.append(f"❌ {t}: Failed to fetch ({str(e)[:50]})")
            continue

    if failed > 0:
        issues.append(f"⚠️ {failed}/{len(ticker_list)} peers failed to load. Check ticker symbols.")

    if not rows:
        issues.append("❌ No peer data retrieved. All tickers failed.")
        return None, issues

    return pd.DataFrame(rows), issues

# ==================== SENSITIVITY ANALYSIS ====================

def create_sensitivity_table(
    base_fcf: float,
    base_wacc: float,
    base_growth: float,
    net_debt: float,
    shares: float,
    forecast_years: int = 5
) -> pd.DataFrame:
    """Create 2D sensitivity table for DCF valuation."""
    wacc_range = np.linspace(
        max(base_wacc - 0.03, MIN_WACC), 
        min(base_wacc + 0.03, MAX_WACC), 
        7
    )
    growth_range = np.linspace(
        max(base_growth - 0.03, 0.01),
        min(base_growth + 0.03, MAX_TERMINAL_GROWTH_INDIA),
        7
    )

    results = []
    for tg in growth_range:
        row = []
        for w in wacc_range:
            if w <= tg:
                row.append(0)
                continue

            fcf_proj = [base_fcf * ((1 + base_growth) ** i) for i in range(1, forecast_years + 1)]
            tv = (fcf_proj[-1] * (1 + tg)) / (w - tg)
            pv_fcf = sum([f / ((1 + w) ** y) for f, y in zip(fcf_proj, range(1, forecast_years + 1))])
            pv_tv = tv / ((1 + w) ** forecast_years)
            eq_val = pv_fcf + pv_tv - net_debt
            imp_price = safe_div(eq_val, shares)
            row.append(imp_price)
        results.append(row)

    df = pd.DataFrame(
        results,
        columns=[f"{w:.1%}" for w in wacc_range],
        index=[f"{g:.1%}" for g in growth_range]
    )
    df.index.name = "Terminal Growth →"
    df.columns.name = "WACC →"
    return df

# ==================== EXCEL EXPORT ====================

def create_professional_dcf_excel(
    workbook, data, years, fcf_proj, wacc, term_growth, pv_tv,
    equity_value, implied_price, tax_rate, growth_rate, current_price
):
    """Create formatted DCF model in Excel matching IB standards."""
    
    ws = workbook.add_worksheet("DCF_Valuation")

    # Formats
    title_fmt = workbook.add_format({"bold": True, "bg_color": "#D9D9D9", "border": 1, "align": "left"})
    section_title_fmt = workbook.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1, "align": "left"})
    header_fmt = workbook.add_format({"bold": True, "border": 1, "align": "center", "bg_color": "#F2F2F2"})
    row_lbl_fmt = workbook.add_format({"border": 1, "align": "left"})
    num_fmt = workbook.add_format({"border": 1, "num_format": "#,##0.00"})
    num0_fmt = workbook.add_format({"border": 1, "num_format": "#,##0"})
    pct_fmt = workbook.add_format({"border": 1, "num_format": "0.0%"})
    bold_num_fmt = workbook.add_format({"border": 1, "num_format": "#,##0.00", "bold": True})
    bold_pct_fmt = workbook.add_format({"border": 1, "num_format": "0.0%", "bold": True})

    # Convert to Cr
    ebit_base_cr = data["ebit"] / 1e7
    fcf_cr = [f / 1e7 for f in fcf_proj]
    pv_tv_cr = pv_tv / 1e7
    cash_cr = data["cash"] / 1e7
    debt_cr = data["debt"] / 1e7

    n = len(years)
    ebit_list = [ebit_base_cr * ((1 + growth_rate) ** i) for i in range(n)]
    ebit_after_tax = [e * (1 - tax_rate) for e in ebit_list]

    reinvest_rate = []
    for ea, f in zip(ebit_after_tax, fcf_cr):
        rr = 1 - (f / ea) if ea != 0 else 0.0
        reinvest_rate.append(rr)

    mid_year = [y - 0.5 for y in years]
    disc_factor = [1 / ((1 + wacc) ** t) if (1 + wacc) != 0 else 1.0 for t in mid_year]
    pv_fcf_cr = [f * d for f, d in zip(fcf_cr, disc_factor)]
    pv_fcf_total_cr = sum(pv_fcf_cr)
    enterprise_cr = pv_fcf_total_cr + pv_tv_cr
    equity_cr = equity_value / 1e7

    # Section 1: PV of FCFF
    start_row, start_col, last_col = 0, 0, n

    ws.merge_range(start_row, start_col, start_row, last_col, "Calculation of PV of FCFF", title_fmt)
    ws.write(start_row + 1, start_col, "", header_fmt)
    for idx, y in enumerate(years):
        ws.write(start_row + 1, start_col + 1 + idx, f"Mar {25 + y}E", header_fmt)

    row = start_row + 2

    def write_row(label, values, fmt):
        nonlocal row
        ws.write(row, start_col, label, row_lbl_fmt)
        for idx, v in enumerate(values):
            ws.write(row, start_col + 1 + idx, clean_number(v), fmt)
        row += 1

    write_row("EBIT", ebit_list, num_fmt)
    write_row("Tax Rate", [tax_rate] * n, pct_fmt)
    write_row("EBIT(1-T)", ebit_after_tax, num_fmt)
    write_row("Less: Reinvestment Rate", reinvest_rate, pct_fmt)
    write_row("Free Cash Flow to Firm (FCFF)", fcf_cr, num_fmt)
    write_row("Mid Year Convention", mid_year, num_fmt)
    write_row("Discounting Factor", disc_factor, num_fmt)
    write_row("PV of FCFF", pv_fcf_cr, num_fmt)

    # Section 2: Summary
    row += 1
    ws.merge_range(row, start_col, row, start_col + 2, "PV of FCFF", section_title_fmt)
    ws.write(row, start_col + 3, clean_number(pv_fcf_total_cr), bold_num_fmt)
    row += 2

    ws.merge_range(row, start_col, row, start_col + 2, "Expected Growth", row_lbl_fmt)
    ws.write(row, start_col + 3, clean_number(growth_rate), pct_fmt)
    row += 1

    ws.merge_range(row, start_col, row, start_col + 2, "Terminal Growth", row_lbl_fmt)
    ws.write(row, start_col + 3, clean_number(term_growth), pct_fmt)
    row += 1

    ws.merge_range(row, start_col, row, start_col + 2, "WACC", row_lbl_fmt)
    ws.write(row, start_col + 3, clean_number(wacc), pct_fmt)
    row += 2

    # Section 3: Terminal Value
    ws.merge_range(row, start_col, row, last_col, "Calculation of Terminal Value", title_fmt)
    row += 1

    fcf_next_cr = fcf_cr[-1] * (1 + term_growth)

    ws.write(row, start_col, "FCFF(t+1)", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(fcf_next_cr), num_fmt)
    row += 1

    ws.write(row, start_col, "WACC", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(wacc), pct_fmt)
    row += 1

    ws.write(row, start_col, "Terminal Growth", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(term_growth), pct_fmt)
    row += 1

    ws.write(row, start_col, "Terminal Value", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(pv_tv_cr), bold_num_fmt)
    row += 2

    # Section 4: Equity value per share
    ws.merge_range(row, start_col, row, last_col, "Calculation of Equity Value per Share", title_fmt)
    row += 1

    ws.write(row, start_col, "PV of FCFF", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(pv_fcf_total_cr), num_fmt)
    row += 1

    ws.write(row, start_col, "PV of Terminal Value", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(pv_tv_cr), num_fmt)
    row += 1

    ws.write(row, start_col, "Value of Enterprise", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(enterprise_cr), bold_num_fmt)
    row += 1

    ws.write(row, start_col, "Add Cash", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(cash_cr), num_fmt)
    row += 1

    ws.write(row, start_col, "Less Debt", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(debt_cr), num_fmt)
    row += 1

    ws.write(row, start_col, "Value of Equity", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(equity_cr), bold_num_fmt)
    row += 1

    ws.write(row, start_col, "No. of Shares (M)", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(data["shares"] / 1e6), num0_fmt)
    row += 1

    ws.write(row, start_col, "Equity Value Per Share", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(implied_price), bold_num_fmt)
    row += 2

    ws.write(row, start_col, "Current Share Price", row_lbl_fmt)
    ws.write(row, start_col + 1, clean_number(current_price), num_fmt)
    row += 1

    ws.write(row, start_col, "Discount/Premium", row_lbl_fmt)
    disc = (implied_price / current_price) - 1 if current_price > 0 else 0.0
    ws.write(row, start_col + 1, clean_number(disc), bold_pct_fmt)

    ws.set_column(0, 0, 35)
    ws.set_column(1, last_col, 14)

# ==================== VISUALIZATION ====================

def create_valuation_waterfall(
    ev: float, net_debt: float, equity_value: float, 
    current_price: float, shares: float
) -> go.Figure:
    """Create bridge from EV to Equity Value to Current Price."""
    implied_mcap = equity_value
    current_mcap = current_price * shares

    fig = go.Figure(go.Waterfall(
        name="Valuation Bridge",
        orientation="v",
        measure=["absolute", "relative", "total", "relative", "total"],
        x=["Enterprise<br>Value", "Less:<br>Net Debt", "Equity<br>Value", 
           "vs Current<br>Market Cap", "Implied<br>Upside"],
        textposition="outside",
        text=[f"₹{ev/1e7:.0f}Cr", f"₹{-net_debt/1e7:.0f}Cr", f"₹{implied_mcap/1e7:.0f}Cr",
              f"₹{(implied_mcap-current_mcap)/1e7:.0f}Cr", 
              f"{((implied_mcap/current_mcap)-1)*100:.1f}%"],
        y=[ev, -net_debt, None, implied_mcap - current_mcap, None],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="DCF Valuation Bridge",
        showlegend=False,
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,20,0.5)"
    )
    return fig

# ==================== VALIDATION ====================

def validate_dcf_assumptions(
    wacc: float, term_growth: float, fcf: float, fcf_margin: float
) -> List[str]:
    """Validate DCF inputs and return list of issues."""
    issues: List[str] = []
    
    if wacc <= term_growth:
        issues.append(
            f"🚨 CRITICAL: WACC ({wacc:.1%}) ≤ Terminal Growth ({term_growth:.1%}). "
            "This breaks Gordon Growth Model mathematics."
        )
    
    if term_growth > MAX_TERMINAL_GROWTH_INDIA:
        issues.append(
            f"⚠️ Terminal Growth ({term_growth:.1%}) > {MAX_TERMINAL_GROWTH_INDIA:.1%} "
            "(India GDP+Inflation). Long-term growth above economy is unrealistic."
        )
    
    if wacc < MIN_WACC:
        issues.append(f"⚠️ WACC ({wacc:.1%}) is very low. Check Rf and Beta inputs.")
    elif wacc > MAX_WACC:
        issues.append(f"⚠️ WACC ({wacc:.1%}) is very high. Company may be extremely risky.")
    
    if fcf <= 0:
        issues.append(f"🚨 CRITICAL: Negative FCF (₹{fcf/1e7:,.0f} Cr). DCF assumes positive future cash flows.")
    
    if fcf_margin < 0:
        issues.append("⚠️ Negative FCF margin. Aggressive to project strong growth from here.")
    
    return issues

def get_wacc_interpretation(wacc: float) -> Tuple[str, str]:
    """Get interpretation of WACC level."""
    if wacc < WACC_LOW_THRESHOLD:
        return "info", f"💡 Low WACC ({wacc:.1%}) suggests stable, mature company or low risk environment."
    elif wacc > WACC_HIGH_THRESHOLD:
        return "warning", f"⚠️ High WACC ({wacc:.1%}) suggests high-growth, high-risk, or distressed company."
    else:
        return "success", f"✅ WACC ({wacc:.1%}) is in reasonable range for Indian equities."

# ==================== MAIN APPLICATION ====================

def main():
    """Main application entry point."""
    
    # ========== SIDEBAR ==========
    st.sidebar.warning(
        "⚠️ **PROTOTYPE MODE**: Uses Yahoo Finance public API for demonstration. "
        "Data for Indian mid-caps may be delayed or require manual overrides. "
        "Production roadmap: Designed to swap for Bloomberg/FactSet API."
    )

    st.sidebar.title("⚙️ Valuation Controls")
    ticker_input = st.sidebar.text_input(
        "Ticker Symbol", 
        value="RELIANCE", 
        help="Enter NSE ticker (with or without .NS)"
    )
    clean_ticker = ticker_input.upper().replace(".NS", "").strip()
    full_ticker = f"{clean_ticker}.NS"

    # Load data
    with st.spinner(f"📊 Loading {clean_ticker} data..."):
        data_payload, dq_tracker = load_ticker_payload(full_ticker)
        rf_rate = get_risk_free_rate()

    if not data_payload or dq_tracker.has_critical_errors():
        st.error("❌ **Cannot proceed with valuation due to critical data issues.**")
        dq_tracker.render()
        st.stop()

    # Scenario Manager
    st.sidebar.divider()
    st.sidebar.subheader("📊 Scenario Manager")
    st.sidebar.caption("Pre-configured assumption sets for different market conditions")
    scenario = st.sidebar.radio("Select Scenario", ["Base Case", "Bear Case", "Bull Case"], index=0)

    scenario_params = {
        "Base Case": {"growth": 12.0, "term_growth": 4.0, "wacc_adj": 0.0},
        "Bear Case": {"growth": 6.0, "term_growth": 3.0, "wacc_adj": 0.015},
        "Bull Case": {"growth": 18.0, "term_growth": 5.0, "wacc_adj": -0.01}
    }

    if clean_ticker in SECTOR_MAP:
        st.sidebar.info(
            f"💡 **Auto-peers:** Using sector map for {clean_ticker}. "
            "Override in Comps tab if needed."
        )

    # God Mode
    st.sidebar.divider()
    with st.sidebar.expander("🔧 God Mode (Override Data)"):
        st.caption("⚠️ Manual overrides – use when yfinance data is incorrect")
        ov_price = st.number_input("Price (₹)", value=float(data_payload["price"]))
        ov_shares = st.number_input("Shares (M)", value=float(data_payload["shares"] / 1e6)) * 1e6
        ov_ebitda = st.number_input("EBITDA (Cr)", value=float(data_payload["ebitda"] / 1e7)) * 1e7
        ov_fcf = st.number_input("FCF (Cr)", value=float(data_payload["fcf"] / 1e7)) * 1e7
        ov_debt = st.number_input("Total Debt (Cr)", value=float(data_payload["debt"] / 1e7)) * 1e7
        ov_cash = st.number_input("Cash (Cr)", value=float(data_payload["cash"] / 1e7)) * 1e7

        if st.button("Apply Overrides"):
            data_payload.update({
                "price": ov_price, "shares": ov_shares, "mcap": ov_price * ov_shares,
                "ebitda": ov_ebitda, "fcf": ov_fcf, "debt": ov_debt, "cash": ov_cash,
                "net_debt": ov_debt - ov_cash,
                "ev": (ov_price * ov_shares) + (ov_debt - ov_cash)
            })
            dq_tracker.add_assumption("🔧 God Mode: Manual data overrides applied")
            st.success("✅ Overrides applied!")

    # ========== HEADER ==========
    st.title(f"🏛️ {clean_ticker} - Investment Banking Valuation Workbench")
    st.caption(
        f"**Scenario:** {scenario} | **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"**Data Source:** Yahoo Finance"
    )

    dq_tracker.render()
    st.divider()

    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Price", f"₹{data_payload['price']:,.2f}")
    col2.metric("Market Cap", f"₹{data_payload['mcap']/1e7:,.0f} Cr")
    col3.metric("Enterprise Value", f"₹{data_payload['ev']/1e7:,.0f} Cr")
    col4.metric("EV/EBITDA", f"{safe_div(data_payload['ev'], data_payload['ebitda']):.2f}x")
    col5.metric("Net Debt", f"₹{data_payload['net_debt']/1e7:,.0f} Cr")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Revenue (TTM)", f"₹{data_payload['revenue']/1e7:,.0f} Cr")
    col2.metric("EBITDA Margin", f"{data_payload['ebitda_margin']*100:.1f}%")
    col3.metric("FCF Margin", f"{data_payload['fcf_margin']*100:.1f}%")
    col4.metric("ROCE", f"{data_payload['roce']*100:.1f}%")
    col5.metric("Net Margin", f"{data_payload['net_margin']*100:.1f}%")

    st.divider()

    # ========== TABS ==========
    tab_wacc, tab_dcf, tab_sens, tab_comps, tab_summ = st.tabs([
        "🧮 WACC Builder", "📉 DCF Model", "🎯 Sensitivity", 
        "⚖️ Comps Analysis", "🏆 Valuation Summary"
    ])

    # ========== TAB 1: WACC ==========
    with tab_wacc:
        st.info(
            "**💡 Objective:** Calculate the Weighted Average Cost of Capital (WACC) – "
            "the discount rate used in DCF valuation."
        )

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Cost of Equity (CAPM)")
            rf = st.number_input(
                "Risk Free Rate (%)", 
                value=rf_rate * 100, 
                step=0.1,
                help="10Y India Govt Bond Yield"
            ) / 100
            erp = st.number_input(
                "Equity Risk Premium (%)", 
                value=5.0, 
                step=0.5,
                help="Expected market return above risk-free"
            ) / 100
            tax = st.number_input(
                "Tax Rate (%)", 
                value=25.0, 
                step=1.0,
                help="Corporate tax rate"
            ) / 100

            st.divider()
            st.write("**Beta Calculation**")
            beta_method = st.radio("Method", ["Regression", "Manual Input"])

            if beta_method == "Manual Input":
                levered_beta = st.number_input("Levered Beta", value=1.0, step=0.1)
                r_squared = 0.0
                fig_beta = None
                beta_issues: List[str] = []
            else:
                period = st.selectbox(
                    "Regression Period",
                    ["6mo", "1y", "2y", "5y"],
                    index=2,
                    help="Longer period = more stable but less recent"
                )
                
                interval = st.radio(
                    "Data Frequency",
                    ["Daily (1d)", "Weekly (1wk)"],
                    index=1,
                    help="Weekly data loads faster, daily provides more precision"
                )
                interval_code = "1wk" if "Weekly" in interval else "1d"
                
                reg_beta, r_squared, fig_beta, msg, beta_issues = calculate_beta_regression(
                    full_ticker, "^NSEI", period, interval_code
                )

                use_blume = st.checkbox(
                    "Apply Blume Adjustment",
                    value=True,
                    help="βadj = (2/3)β + (1/3) – mean-reverts beta toward 1.0"
                )
                levered_beta = (0.67 * reg_beta + 0.33) if use_blume else reg_beta

                st.metric("Raw Beta", f"{reg_beta:.3f}")
                st.metric("Adjusted Beta", f"{levered_beta:.3f}")
                st.metric("R² (Fit Quality)", f"{r_squared:.3f}")

                if beta_issues:
                    for issue in beta_issues:
                        st.warning(issue)

            if levered_beta > 2.5 or levered_beta < 0.2:
                st.warning("⚠️ Unusual Beta. Verify ticker or consider illiquidity issues.")

            unlevered_beta = calculate_unlevered_beta(
                levered_beta, data_payload["debt"], data_payload["mcap"], tax
            )
            st.metric(
                "Unlevered Beta",
                f"{unlevered_beta:.3f}",
                help="βu = βl / [1 + (1 - T)(D/E)]"
            )

            st.divider()
            st.subheader("Cost of Debt")

            icr = safe_div(data_payload["ebitda"], data_payload["interest"], 100)
            st.metric("Interest Coverage Ratio", f"{icr:.2f}x", help="EBITDA / Interest Expense")

            if icr > 8:
                spread, rating = 0.010, "AAA/AA (Very Safe)"
            elif icr > 4:
                spread, rating = 0.025, "A/BBB (Investment Grade)"
            elif icr > 1.5:
                spread, rating = 0.045, "BB/B (Speculative)"
            else:
                spread, rating = 0.080, "CCC or lower (Distressed)"

            st.write(f"**Implied Credit Rating:** {rating}")

            manual_kd = st.checkbox(
                "Override Cost of Debt",
                value=False,
                help="Use if auto-calculated spread seems unreasonable"
            )

            if manual_kd:
                spread_adj = st.number_input(
                    "Credit Spread (bps)", 
                    value=int(spread * 10000), 
                    step=10
                ) / 10000
                st.caption(
                    "💡 Typical: 10–25bps (AAA), 25–50bps (A/BBB), "
                    "50–100bps (BB/B), 100+bps (High Yield)"
                )
            else:
                spread_adj = spread

            rd_pretax = rf + spread_adj
            rd_aftertax = rd_pretax * (1 - tax)

            st.metric("Kd (Pre-tax)", f"{rd_pretax:.2%}")
            st.metric("Kd (After-tax)", f"{rd_aftertax:.2%}")

        with col_right:
            st.subheader("WACC Calculation")

            ke = rf + (levered_beta * erp)

            E = data_payload["mcap"]
            D = data_payload["debt"]
            V = E + D if (E + D) > 0 else 1

            weight_e = E / V
            weight_d = D / V

            if D == 0 and data_payload["interest"] > 0:
                st.warning("⚠️ Debt is zero but interest exists. Check balance sheet data.")

            wacc = (weight_e * ke) + (weight_d * rd_aftertax)

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
                        padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;">
                <div style="font-size: 0.9rem; color: #a5d6a7; text-transform: uppercase;
                            letter-spacing: 2px; margin-bottom: 10px;">
                    Weighted Average Cost of Capital
                </div>
                <div style="font-size: 3.5rem; font-weight: 800; color: #ffffff; margin: 15px 0;">
                    {wacc:.2%}
                </div>
                <div style="font-size: 1rem; color: #c8e6c9; margin-top: 15px;">
                    Ke: {ke:.2%} | Kd(AT): {rd_aftertax:.2%} | β: {levered_beta:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption("**Formula:** WACC = (E/V × Re) + (D/V × Rd × (1 - T))")

            # WACC Interpretation
            interp_type, interp_msg = get_wacc_interpretation(wacc)
            if interp_type == "info":
                st.info(interp_msg)
            elif interp_type == "warning":
                st.warning(interp_msg)
            else:
                st.success(interp_msg)

            with st.expander("See Calculation Logic"):
                st.latex(r"WACC = \frac{E}{V} \times R_e + \frac{D}{V} \times R_d \times (1 - T)")
                st.markdown(f"""
                **Inputs used:**
                - Equity Weight (E/V): **{weight_e:.2%}**
                - Cost of Equity (Re): **{ke:.2%}**
                - Debt Weight (D/V): **{weight_d:.2%}**
                - Cost of Debt (Rd): **{rd_pretax:.2%}**
                - Tax Rate (T): **{tax:.2%}**
                """)

            if fig_beta:
                st.plotly_chart(fig_beta, use_container_width=True)

            st.subheader("Capital Structure (Market Values)")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Equity Weight", f"{weight_e:.1%}")
                st.metric("Market Cap", f"₹{E/1e7:,.0f} Cr")
            with c2:
                st.metric("Debt Weight", f"{weight_d:.1%}")
                st.metric("Total Debt", f"₹{D/1e7:,.0f} Cr")

            fig_cap = go.Figure(data=[go.Pie(
                labels=["Equity", "Debt"],
                values=[E, D],
                hole=0.4,
                marker_colors=["#4caf50", "#ff6f00"]
            )])
            fig_cap.update_layout(
                title="Capital Structure",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True
            )
            st.plotly_chart(fig_cap, use_container_width=True)

    # ========== TAB 2: DCF ==========
    with tab_dcf:
        st.info(
            "**💡 Objective:** Project FCF, discount using WACC, and derive intrinsic equity value per share."
        )

        col_assumptions, col_results = st.columns([1, 2])

        with col_assumptions:
            st.subheader("📋 DCF Assumptions")

            default_growth = scenario_params[scenario]["growth"]
            default_tg = scenario_params[scenario]["term_growth"]
            wacc_adjustment = scenario_params[scenario]["wacc_adj"]

            forecast_years = st.slider(
                "Forecast Period (Years)", 
                3, 10, 5,
                help="Years of explicit FCF projection"
            )
            growth_rate = st.number_input(
                "Growth Rate (Explicit Period, %)",
                value=default_growth,
                step=1.0
            ) / 100
            terminal_growth = st.number_input(
                "Terminal Growth (%)",
                value=default_tg,
                step=0.5,
                help="Perpetual growth rate after explicit period"
            ) / 100
            wacc_dcf = st.number_input(
                "WACC (%)", 
                value=(wacc + wacc_adjustment) * 100, 
                step=0.1
            ) / 100

            use_midyear = st.checkbox(
                "Mid-year Convention",
                value=True,
                help="Assumes cash flows occur mid-year (more realistic)"
            )

            st.divider()
            st.write("**Starting Metrics (TTM)**")
            st.metric("Base FCF", f"₹{data_payload['fcf']/1e7:,.0f} Cr")
            st.metric("EBITDA", f"₹{data_payload['ebitda']/1e7:,.0f} Cr")
            st.metric("Net Debt", f"₹{data_payload['net_debt']/1e7:,.0f} Cr")

            # Validation
            dcf_issues = validate_dcf_assumptions(
                wacc_dcf, terminal_growth, 
                data_payload["fcf"], data_payload["fcf_margin"]
            )
            
            if dcf_issues:
                st.divider()
                st.error("**⚠️ DCF Validation Issues**")
                for issue in dcf_issues:
                    st.warning(issue)
            
            # Assumption Summary
            with st.expander("📋 Assumption Summary (for documentation)"):
                st.markdown(f"""
                - **Explicit Growth:** {growth_rate:.1%} for {forecast_years} years
                - **Terminal Growth:** {terminal_growth:.1%} (perpetuity)
                - **WACC:** {wacc_dcf:.2%}
                - **Convention:** {'Mid-year' if use_midyear else 'Year-end'}
                - **Tax Rate:** {tax:.1%}
                - **Base FCF (TTM):** ₹{data_payload['fcf']/1e7:,.0f} Cr
                """)

        with col_results:
            st.subheader("📊 DCF Valuation Results")

            if wacc_dcf <= terminal_growth:
                st.error("🚨 **CANNOT CALCULATE:** WACC ≤ Terminal Growth. Adjust assumptions above.")
                st.stop()

            years = list(range(1, forecast_years + 1))
            fcf_projections = [data_payload["fcf"] * ((1 + growth_rate) ** i) for i in years]

            terminal_fcf = fcf_projections[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (wacc_dcf - terminal_growth)

            if use_midyear:
                discount_factors = [1 / ((1 + wacc_dcf) ** (y - 0.5)) for y in years]
                tv_discount = 1 / ((1 + wacc_dcf) ** (forecast_years - 0.5))
            else:
                discount_factors = [1 / ((1 + wacc_dcf) ** y) for y in years]
                tv_discount = 1 / ((1 + wacc_dcf) ** forecast_years)

            pv_fcf_explicit = sum([fcf * df for fcf, df in zip(fcf_projections, discount_factors)])
            pv_terminal = terminal_value * tv_discount

            enterprise_value = pv_fcf_explicit + pv_terminal
            equity_value = enterprise_value - data_payload["net_debt"]
            implied_price = safe_div(equity_value, data_payload["shares"])

            current_price = data_payload["price"]
            upside = ((implied_price / current_price) - 1) * 100 if current_price > 0 else 0

           
           # Display Results
            rc1, rc2, rc3 = st.columns(3)
            
            # Column 1: Enterprise Value
            with rc1:
                st.markdown(f"""
                <div style="background:#1a237e;padding:15px;border-radius:8px;text-align:center;">
                    <div style="font-size:0.75rem;color:#9fa8da;">Enterprise Value</div>
                    <div style="font-size:1.8rem;font-weight:700;color:#fff;">₹{enterprise_value/1e7:,.0f} Cr</div>
                    <div style="font-size:0.7rem;color:#9fa8da;margin-top:5px;">PV Explicit + Terminal</div>
                </div>
                """, unsafe_allow_html=True)

            # Column 2: Equity Value (THIS WAS THE MISSING PART)
            with rc2:
                st.markdown(f"""
                <div style="background:#1b5e20;padding:15px;border-radius:8px;text-align:center;">
                    <div style="font-size:0.75rem;color:#a5d6a7;">Equity Value</div>
                    <div style="font-size:1.8rem;font-weight:700;color:#fff;">₹{equity_value/1e7:,.0f} Cr</div>
                    <div style="font-size:0.7rem;color:#a5d6a7;margin-top:5px;">EV - Net Debt</div>
                </div>
                """, unsafe_allow_html=True)

            # Column 3: Implied Price
            with rc3:
                c = "#4caf50" if upside > 0 else "#f44336"
                st.markdown(f"""
                <div style="background:{c};padding:15px;border-radius:8px;text-align:center;">
                    <div style="font-size:0.75rem;color:#fff;">Implied Price</div>
                    <div style="font-size:1.8rem;font-weight:700;color:#fff;">₹{implied_price:,.0f}</div>
                    <div style="font-size:0.85rem;color:#fff;margin-top:5px;">
                        {upside:+.1f}% vs Current
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.subheader("Free Cash Flow Forecast")

            fcf_display = [data_payload["fcf"] / 1e7] + [f / 1e7 for f in fcf_projections] + [terminal_fcf / 1e7]
            pv_display = [np.nan] + [(f * df) / 1e7 for f, df in zip(fcf_projections, discount_factors)] + [pv_terminal / 1e7]

            df_fcf = pd.DataFrame({
                "Year": ["TTM"] + [f"Year {y}" for y in years] + ["Terminal"],
                "FCF (Cr)": fcf_display,
                "Growth": [np.nan] + [growth_rate for _ in years] + [terminal_growth],
                "Discount Factor": [np.nan] + discount_factors + [tv_discount],
                "PV (Cr)": pv_display
            })

            st.dataframe(
                df_fcf.style.format({
                    "FCF (Cr)": "₹{:,.0f}",
                    "Growth": "{:.1%}",
                    "Discount Factor": "{:.4f}",
                    "PV (Cr)": "₹{:,.0f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Valuation Bridge")
            st.caption("Shows how we get from Enterprise Value to Implied Share Price")
            fig_waterfall = create_valuation_waterfall(
                enterprise_value, data_payload["net_debt"], 
                equity_value, current_price, data_payload["shares"]
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

            st.divider()

            # Excel Export
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                workbook = writer.book

                create_professional_dcf_excel(
                    workbook=workbook, data=data_payload, years=years,
                    fcf_proj=fcf_projections, wacc=wacc_dcf, term_growth=terminal_growth,
                    pv_tv=pv_terminal, equity_value=equity_value, implied_price=implied_price,
                    tax_rate=tax, growth_rate=growth_rate, current_price=current_price
                )

                df_summary_export = pd.DataFrame({
                    "Item": ["PV of FCFF (Cr)", "PV of Terminal Value (Cr)", "Enterprise Value (Cr)",
                            "Net Debt (Cr)", "Equity Value (Cr)", "Implied Price", "Current Price", "Upside (%)"],
                    "Value": [pv_fcf_explicit / 1e7, pv_terminal / 1e7, enterprise_value / 1e7,
                             data_payload["net_debt"] / 1e7, equity_value / 1e7, implied_price, current_price, upside]
                })
                df_summary_export.to_excel(writer, sheet_name="Summary", index=False)

                df_assumptions = pd.DataFrame({
                    "Parameter": ["Ticker", "Scenario", "Forecast Years", "Growth Rate (Explicit)",
                                 "Terminal Growth", "WACC (DCF)", "Tax Rate", "Risk Free", "Beta", "Date"],
                    "Value": [clean_ticker, scenario, forecast_years, growth_rate, terminal_growth,
                             wacc_dcf, tax, rf, levered_beta, datetime.now().strftime("%Y-%m-%d")]
                })
                df_assumptions.to_excel(writer, sheet_name="Assumptions", index=False)

            st.success("✅ Model ready for export")
            st.caption("Generates fully formatted .xlsx file with IB-standard layout for client deliverables.")
            st.download_button(
                "📥 Export Audit-Ready Excel Model",
                data=buffer.getvalue(),
                file_name=f"{clean_ticker}_DCF_{scenario.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # ========== TAB 3: SENSITIVITY ==========
    with tab_sens:
        st.info(
            "**💡 Objective:** Test robustness of DCF valuation across different WACC and Terminal Growth "
            "combinations to understand valuation range."
        )
        st.subheader("🎯 DCF Sensitivity Analysis")

        sc1, sc2 = st.columns([2, 1])
        
        with sc2:
            st.write("**Reference Values**")
            st.metric("Base WACC", f"{wacc_dcf:.2%}")
            st.metric("Base Terminal Growth", f"{terminal_growth:.2%}")
            st.metric("Current Price", f"₹{current_price:,.0f}")

        with sc1:
            sens_table = create_sensitivity_table(
                data_payload["fcf"], wacc_dcf, growth_rate,
                data_payload["net_debt"], data_payload["shares"], forecast_years
            )

            sens_for_display = sens_table.replace(0, np.nan)

            def highlight_current(val):
                if pd.isna(val):
                    return "background-color: #b71c1c; color: white"
                diff_pct = abs((val - current_price) / current_price)
                if diff_pct < 0.05:
                    return "background-color: #1b5e20; font-weight: bold"
                elif diff_pct < 0.15:
                    return "background-color: #2e7d32"
                return ""

            styled_sens = sens_for_display.style.format("₹{:,.0f}").applymap(highlight_current)
            st.dataframe(styled_sens, use_container_width=True)

            st.caption(
                "💡 **Green cells:** Within 5–15% of current price | "
                "**Red cells:** Invalid (WACC ≤ Terminal Growth)"
            )
            
            # Export sensitivity table
            csv_buffer = sens_table.to_csv()
            st.download_button(
                "📥 Download Sensitivity Table (CSV)",
                data=csv_buffer,
                file_name=f"{clean_ticker}_Sensitivity_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        st.divider()
        st.subheader("Valuation Heatmap")

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=sens_table.values,
            x=sens_table.columns,
            y=sens_table.index,
            colorscale="RdYlGn",
            text=sens_table.values,
            texttemplate="₹%{text:,.0f}",
            textfont={"size": 9},
            hovertemplate="WACC: %{x}<br>Term Growth: %{y}<br>Price: ₹%{z:,.0f}<extra></extra>"
        ))
        fig_heatmap.update_layout(
            title=f"DCF Implied Price Sensitivity ({scenario})",
            xaxis_title="WACC →",
            yaxis_title="Terminal Growth →",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,20,0.5)"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ========== TAB 4: COMPS ==========
    with tab_comps:
        st.info(
            "**💡 Objective:** Value the company using trading multiples from comparable peers "
            "to triangulate intrinsic value vs. DCF."
        )
        st.subheader("⚖️ Comparable Companies Analysis")

        default_peers = ", ".join(SECTOR_MAP.get(clean_ticker, ["TCS.NS", "INFY.NS"]))

        peer_input = st.text_area(
            "Enter Peer Tickers (comma-separated)",
            value=default_peers,
            height=80,
            help="NSE tickers with or without .NS. Auto-populated from sector map."
        )

        if len(clean_peer_list(peer_input)) > MAX_PEERS:
            st.warning(
                f"⚠️ You entered {len(clean_peer_list(peer_input))} peers. "
                f"This may be slow. Consider reducing to top {MAX_PEERS}."
            )

        cc1, cc2 = st.columns([3, 1])
        with cc2:
            run_comps = st.button("🔍 Analyze Peers", type="primary", use_container_width=True)

        if run_comps or "comps_data" in st.session_state:
            peer_list = clean_peer_list(peer_input)
            peer_list = [p for p in peer_list if p != full_ticker]
            
            # Progress bar for peer fetching
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"Fetching data for {len(peer_list)} peers...")
            
            df_peers, peer_issues = get_detailed_peers(peer_list)
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            if peer_issues:
                with st.expander("⚠️ Peer Data Issues"):
                    for issue in peer_issues:
                        st.warning(issue)

            if df_peers is not None and not df_peers.empty:
                st.session_state["comps_data"] = df_peers

                st.dataframe(
                    df_peers.style.format({
                        "Price": "₹{:,.0f}",
                        "Market Cap (Cr)": "{:,.0f}",
                        "EV (Cr)": "{:,.0f}",
                        "Revenue (Cr)": "{:,.0f}",
                        "EBITDA (Cr)": "{:,.0f}",
                        "EV/Revenue": "{:.2f}x",
                        "EV/EBITDA": "{:.1f}x",
                        "P/E": "{:.1f}x",
                        "EBITDA Margin": "{:.1%}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                st.divider()
                st.subheader("Trading Multiple Statistics")

                s1, s2 = st.columns(2)
                
                with s1:
                    ev_ebitda_valid = df_peers["EV/EBITDA"].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(ev_ebitda_valid) > 0:
                        ev_stats = {
                            "Min": ev_ebitda_valid.min(),
                            "25th Percentile": ev_ebitda_valid.quantile(0.25),
                            "Median": ev_ebitda_valid.median(),
                            "Mean": ev_ebitda_valid.mean(),
                            "75th Percentile": ev_ebitda_valid.quantile(0.75),
                            "Max": ev_ebitda_valid.max()
                        }
                        df_ev_stats = pd.DataFrame(list(ev_stats.items()), columns=["Statistic", "EV/EBITDA"])
                        st.write("**EV/EBITDA Multiples**")
                        st.dataframe(
                            df_ev_stats.style.format({"EV/EBITDA": "{:.2f}x"}),
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.warning("No valid EV/EBITDA data from peers")

                with s2:
                    pe_valid = df_peers["P/E"].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(pe_valid) > 0:
                        pe_stats = {
                            "Min": pe_valid.min(),
                            "25th Percentile": pe_valid.quantile(0.25),
                            "Median": pe_valid.median(),
                            "Mean": pe_valid.mean(),
                            "75th Percentile": pe_valid.quantile(0.75),
                            "Max": pe_valid.max()
                        }
                        df_pe_stats = pd.DataFrame(list(pe_stats.items()), columns=["Statistic", "P/E"])
                        st.write("**P/E Multiples**")
                        st.dataframe(
                            df_pe_stats.style.format({"P/E": "{:.2f}x"}),
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.warning("No valid P/E data from peers")

                st.divider()
                st.subheader("Implied Valuation from Peer Multiples")
                st.caption("Apply peer multiples to target company metrics to estimate fair value")

                if len(ev_ebitda_valid) > 0:
                    percentiles = {
                        "Bear (25th %ile)": ev_ebitda_valid.quantile(0.25),
                        "Base (Median)": ev_ebitda_valid.median(),
                        "Bull (75th %ile)": ev_ebitda_valid.quantile(0.75)
                    }

                    def calc_price_from_multiple(multiple):
                        implied_ev = data_payload["ebitda"] * multiple
                        implied_equity = implied_ev - data_payload["net_debt"]
                        return safe_div(implied_equity, data_payload["shares"])

                    vc1, vc2, vc3 = st.columns(3)
                    results = {}
                    
                    for idx, (label, mult) in enumerate(percentiles.items()):
                        price_mult = calc_price_from_multiple(mult)
                        results[label] = price_mult
                        up_mult = ((price_mult / data_payload["price"]) - 1) * 100

                        col = [vc1, vc2, vc3][idx]
                        color = "#c62828" if "Bear" in label else ("#1b5e20" if "Bull" in label else "#1565c0")
                        
                        col.markdown(f"""
                        <div style="background:{color};padding:20px;border-radius:10px;text-align:center;">
                            <div style="font-size:0.8rem;color:#fff;opacity:0.9;">{label}</div>
                            <div style="font-size:2rem;font-weight:700;color:#fff;margin:10px 0;">
                                ₹{price_mult:,.0f}
                            </div>
                            <div style="font-size:0.85rem;color:#fff;opacity:0.8;">
                                {mult:.2f}x EV/EBITDA<br>
                                {up_mult:+.1f}% vs Current
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.session_state["comps_results"] = results

                    st.divider()
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=ev_ebitda_valid,
                        name="EV/EBITDA",
                        marker_color="#2196f3",
                        boxmean="sd"
                    ))
                    fig_box.update_layout(
                        title="Peer EV/EBITDA Distribution",
                        yaxis_title="EV/EBITDA Multiple",
                        height=400,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(20,20,20,0.5)",
                        showlegend=False
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

            else:
                st.error("❌ No peer data found. Please check ticker symbols.")

    # ========== TAB 5: SUMMARY ==========
    with tab_summ:
        st.info(
            "**💡 Objective:** Synthesize DCF and Comps valuations into a Football Field Chart "
            "to guide the investment recommendation."
        )
        st.subheader("🏆 Valuation Summary - Football Field Chart")

        has_dcf = "implied_price" in locals() and implied_price > 0
        has_comps = "comps_results" in st.session_state

        if has_dcf or has_comps:
            fig_football = go.Figure()

            if has_dcf:
                dcf_low = implied_price * 0.85
                dcf_high = implied_price * 1.15

                fig_football.add_trace(go.Bar(
                    y=[f"DCF<br>({scenario})"],
                    x=[dcf_high - dcf_low],
                    base=[dcf_low],
                    orientation="h",
                    name="DCF",
                    marker_color="#4caf50",
                    hovertemplate=f"<b>DCF Range</b><br>Low: ₹{dcf_low:,.0f}<br>High: ₹{dcf_high:,.0f}<br>Mid: ₹{implied_price:,.0f}<extra></extra>"
                ))

                fig_football.add_trace(go.Scatter(
                    y=[f"DCF<br>({scenario})"],
                    x=[implied_price],
                    mode="markers",
                    marker=dict(size=15, color="white", symbol="diamond", line=dict(color="#4caf50", width=2)),
                    name="DCF Midpoint",
                    showlegend=False,
                    hovertemplate=f"<b>DCF</b><br>₹{implied_price:,.0f}<extra></extra>"
                ))

            if has_comps:
                comps = st.session_state["comps_results"]
                bear = comps.get("Bear (25th %ile)", 0)
                base = comps.get("Base (Median)", 0)
                bull = comps.get("Bull (75th %ile)", 0)

                if bear > 0 and bull > 0:
                    fig_football.add_trace(go.Bar(
                        y=["Trading<br>Comps"],
                        x=[bull - bear],
                        base=[bear],
                        orientation="h",
                        name="Comps",
                        marker_color="#2196f3",
                        hovertemplate=f"<b>Comps Range</b><br>Bear: ₹{bear:,.0f}<br>Base: ₹{base:,.0f}<br>Bull: ₹{bull:,.0f}<extra></extra>"
                    ))

                    fig_football.add_trace(go.Scatter(
                        y=["Trading<br>Comps"],
                        x=[base],
                        mode="markers",
                        marker=dict(size=15, color="white", symbol="diamond", line=dict(color="#2196f3", width=2)),
                        name="Comps Median",
                        showlegend=False,
                        hovertemplate=f"<b>Comps Median</b><br>₹{base:,.0f}<extra></extra>"
                    ))

            fig_football.add_vline(
                x=data_payload["price"],
                line_dash="dash",
                line_color="white",
                line_width=2,
                annotation_text=f"Current: ₹{data_payload['price']:,.0f}",
                annotation_position="top"
            )

            fig_football.update_layout(
                title=f"{clean_ticker} - Valuation Summary",
                xaxis_title="Implied Price (₹)",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(20,20,20,0.5)",
                showlegend=True,
                barmode="overlay",
                xaxis=dict(gridcolor="rgba(128,128,128,0.2)")
            )
            st.plotly_chart(fig_football, use_container_width=True)

            st.divider()
            st.subheader("Valuation Summary Table")

            summary_data = []
            
            if has_dcf:
                summary_data.append({
                    "Method": "DCF",
                    "Low": implied_price * 0.85,
                    "Base": implied_price,
                    "High": implied_price * 1.15,
                    "Upside to Base": ((implied_price / data_payload["price"]) - 1) * 100
                })
            
            if has_comps:
                comps = st.session_state["comps_results"]
                summary_data.append({
                    "Method": "Trading Comps",
                    "Low": comps.get("Bear (25th %ile)", 0),
                    "Base": comps.get("Base (Median)", 0),
                    "High": comps.get("Bull (75th %ile)", 0),
                    "Upside to Base": ((comps.get("Base (Median)", data_payload["price"]) / data_payload["price"]) - 1) * 100
                })

            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                
                if len(summary_data) > 1:
                    blended = {
                        "Method": "Blended Average",
                        "Low": df_summary["Low"].mean(),
                        "Base": df_summary["Base"].mean(),
                        "High": df_summary["High"].mean(),
                        "Upside to Base": ((df_summary["Base"].mean() / data_payload["price"]) - 1) * 100
                    }
                    df_summary = pd.concat([df_summary, pd.DataFrame([blended])], ignore_index=True)

                st.dataframe(
                    df_summary.style.format({
                        "Low": "₹{:,.0f}",
                        "Base": "₹{:,.0f}",
                        "High": "₹{:,.0f}",
                        "Upside to Base": "{:+.1f}%"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                st.divider()
                avg_upside = df_summary[df_summary["Method"] != "Blended Average"]["Upside to Base"].mean()

                if avg_upside > 20:
                    recommendation = "🟢 **STRONG BUY**"
                    color = "#1b5e20"
                    rationale = "Significant upside (>20%) indicates the stock is meaningfully undervalued."
                elif avg_upside > 10:
                    recommendation = "🟢 **BUY**"
                    color = "#2e7d32"
                    rationale = "Attractive upside (10–20%) suggests good risk–reward for accumulation."
                elif avg_upside > -10:
                    recommendation = "🟡 **HOLD**"
                    color = "#f57f17"
                    rationale = "Fair value range (±10%). Hold existing positions, no urgent action."
                elif avg_upside > -20:
                    recommendation = "🔴 **REDUCE**"
                    color = "#d84315"
                    rationale = "Modest downside (10–20%). Consider reducing exposure or tightening stops."
                else:
                    recommendation = "🔴 **SELL**"
                    color = "#b71c1c"
                    rationale = "Significant downside (>20%) suggests the stock is overvalued. Exit positions."

                st.markdown(f"""
                <div style="background:{color};padding:25px;border-radius:12px;text-align:center;margin:20px 0;">
                    <div style="font-size:1rem;color:#fff;opacity:0.9;margin-bottom:10px;">
                        Investment Recommendation
                    </div>
                    <div style="font-size:2.5rem;font-weight:800;color:#fff;">
                        {recommendation}
                    </div>
                    <div style="font-size:1.1rem;color:#fff;opacity:0.9;margin-top:10px;">
                        Average Upside: {avg_upside:+.1f}%
                    </div>
                    <div style="font-size:0.9rem;color:#fff;opacity:0.85;margin-top:15px;font-style:italic;">
                        {rationale}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.divider()
                st.subheader("⚠️ Key Risks & Considerations")

                r1, r2 = st.columns(2)
                
                with r1:
                    st.markdown("**Valuation Risks:**")
                    risks = []
                    if terminal_growth > 0.05:
                        risks.append(f"• High terminal growth ({terminal_growth:.1%}) may be optimistic")
                    if data_payload["fcf_margin"] < 0.05:
                        risks.append(f"• Low FCF margin ({data_payload['fcf_margin']:.1%}) – execution risk")
                    if data_payload["debt"] / data_payload["mcap"] > 1:
                        risks.append("• High leverage (D/E > 1.0) – financial risk")
                    if not risks:
                        risks.append("• No major valuation red flags based on current inputs")
                    for r in risks:
                        st.write(r)

                with r2:
                    st.markdown("**Data Quality:**")
                    data_risks = []
                    if len(dq_tracker.issues) > 0:
                        data_risks.append(f"• {len(dq_tracker.issues)} data quality warning(s)")
                    if len(dq_tracker.assumptions) > 3:
                        data_risks.append(f"• {len(dq_tracker.assumptions)} key assumptions made")
                    if "beta_issues" in locals() and len(beta_issues) > 0:
                        data_risks.append("• Beta regression has quality/fit concerns")
                    if not data_risks:
                        data_risks.append("• Data quality appears reasonable for analysis")
                    for r in data_risks:
                        st.write(r)

        else:
            st.info("👆 **Complete DCF Model and/or Comps Analysis** to generate the Football Field chart.")

    # ========== FOOTER ==========
    st.divider()
    st.caption(f"""
    **⚠️ Disclaimer:** This tool is for educational and analytical purposes only. Not investment advice.
    Always conduct thorough due diligence and consult with financial professionals before making investment decisions.

    **Data Sources:** Yahoo Finance (yfinance) | Risk-Free Rate: India 10Y Govt Bond ({rf_rate:.2%}) | Market Index: NIFTY 50 (^NSEI)

    **Limitations:** yfinance data can be stale, incomplete, or incorrect for Indian stocks. Use God Mode overrides for critical corrections.
    Enterprise Value, EBITDA, and other metrics may be missing for smaller/newer companies. Always cross-verify with official filings.
    """)
    st.caption(f"**Version:** 2.1 Professional | **Last Updated:** {datetime.now().strftime('%Y-%m-%d')} | **Built with:** Streamlit + Python")

# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    main()
 