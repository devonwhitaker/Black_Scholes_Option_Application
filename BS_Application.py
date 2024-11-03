import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import norm
import seaborn as sns
import plotly.express as px
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go

index = ["Curent Price", "Strike Price", "Maturity", "Risk Free", "Volatility"]

def b_scholes(S0, strike, t, r, sigma, type='call'):
    """
    S0: Current Price
    strike: strike price
    t: time to maturity in days
    r: risk-free rate
    sigma: volatility
    type: call or put
    """

    t = t / 365 #allow for days to maturity

    d1 = (np.log(S0 / strike) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))

    if type == 'call':
        price = S0 * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)
    elif type == 'put':
        price = strike * np.exp(-r * t) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("type must be either call or put")
    
    return price

def calc_greeks(S0, strike, t, r, sigma, type='call'):
    t = t / 365
    d1 = (np.log(S0 / strike) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))

    delta = norm.cdf(d1) if type== 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(t))
    vega = S0 * norm.pdf(d1) * np.sqrt(t) / 100
    theta = -((S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(t))) / 365

    return delta, gamma, vega, theta


st.set_page_config(layout="wide")

c1, c2, c3= st.columns(3)
with c1:
    st.title("Inputs:")
    current_price = st.number_input("Current Price:", min_value=0.0, step=0.1, value=30.0)
    strike_price = st.number_input("Strike Price:", min_value=0.0, step=0.1, value=30)
    maturity = st.slider("Maturity (Days):", min_value=0, step=1, value=30.0)
    risk_free = st.number_input("Risk-Free Rate:", min_value=0.00, step=0.01, value=0.06)
    sigma = st.slider("Volatility:", min_value=0.00, step=0.01, value=0.20)

    st.markdown("---\n ***Stock Option Analysis:***")
    
    ticker = st.text_input("*Ticker:*")

    if ticker:

        ticker_info = yf.Ticker(ticker)
        ticker_data = ticker_info.history(period="1y")

        # Store ticker related data if present
        ticker_price = ticker_data['Close'][-1]
        ticker_rets = ticker_data['Close'].pct_change().dropna()
        ticker_sigma = ticker_rets.std() * np.sqrt(252) # Done to annualize the Volatility

        ticker_maturities = ticker_info.options
        
        now = datetime.now()

        daystomat = [(date,(datetime.strptime(date, '%Y-%m-%d') - now).days) for date in ticker_maturities]
        df_dtm = pd.DataFrame(daystomat, columns=["Maturity Date", "Days to Maturity"])
        df_dtm = df_dtm[:8]

        ticker_maturity = st.radio("**Upcoming Option Expiry Dates:**", df_dtm["Days to Maturity"].tolist())

        s_call_price = b_scholes(ticker_price, strike_price, ticker_maturity, risk_free, ticker_sigma)
        s_put_price = b_scholes(ticker_price, strike_price, ticker_maturity, risk_free, ticker_sigma, type='put')

        stock_option = pd.DataFrame(data= [ticker_price, strike_price, ticker_maturity, risk_free, ticker_sigma],index=index)
        stock_option = stock_option.round(2)

call_price = b_scholes(current_price, strike_price, maturity, risk_free, sigma)
put_price = b_scholes(current_price, strike_price, maturity, risk_free, sigma, type='put')

simulated_option = pd.DataFrame(data= [current_price, strike_price, maturity, risk_free, sigma],index=index)
simulated_option = simulated_option.round(2)

with c2:

    st.header("Simulated Option:")
    st.markdown(
        f"""
        <div style="background-color: blue; padding: 5px; border-radius: 10px; margin-bottom: 10px;">
            <h2 style="text-align: center; color: white;">Call Price: ${call_price:.2f}</h2>
        </div>
        <div style="background-color: Red; padding: 5px; border-radius: 10px; margin-bottom: 10px;">
            <h2 style="text-align: center; color: white;">Put Price: ${put_price:.2f}</h2>
        <div>
        """,
        unsafe_allow_html=True
        )

    st.subheader("Option Parameters:")
    for index, value in simulated_option.iterrows():
        st.markdown(
            f"""
            <div style="padding: 8px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 5px; text-align: center;">
                <strong style="margin-right: 10px;">{index}:</strong> <span style="font-weight: normal;">{value[0]}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

with c3:
    if ticker:
        st.header("Stock Option:")
        st.markdown(
            f"""
            <div style="background-color: green; padding: 5px; border-radius: 10px; margin-bottom: 10px;">
                <h2 style="text-align: center; color: white;">Call Price: ${s_call_price:.2f}</h2>
            </div>
            <div style="background-color: orange; padding: 5px; border-radius: 10px; margin-bottom: 10px;">
                <h2 style="text-align: center; color: white;">Put Price: ${s_put_price:.2f}</h2>
            <div>
            """,
            unsafe_allow_html=True
            )

        st.subheader("Option Parameters:")
        for index, value in stock_option.iterrows():
            st.markdown(
                f"""
                <div style="padding: 8px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 5px; text-align: center;">
                    <strong style="margin-right: 10px;">{index}:</strong> <span style="font-weight: normal;">{value[0]}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

with c1:
    if ticker:
    
        st.subheader("Historical Price and Volatility")
        price_fig = px.line(ticker_data, x=ticker_data.index, y="Close", title=f"{ticker} Price Trend")
        st.plotly_chart(price_fig, use_container_width=True)
        
        ticker_data['Rolling Volatility'] = ticker_data['Close'].pct_change().rolling(window=21).std() * np.sqrt(252)
        volatility_fig = px.line(ticker_data, x=ticker_data.index, y="Rolling Volatility", title=f"{ticker} Rolling Volatility (21-Day)")
        st.plotly_chart(volatility_fig, use_container_width=True)


with c2:
    
    c_current_range = np.linspace(current_price*0.5, current_price*1.5, 50)
    c_maturity_range = np.linspace(0, 365, 50)
    c_sigma_range = np.linspace(sigma*0.5, sigma*2, 50)

    c_current, c_vol = np.meshgrid(c_current_range, c_sigma_range)

    call_prices = np.zeros(c_current.shape)
    for j in range(c_sigma_range.shape[0]):
        for i in range(c_current.shape[0]):
            call_prices[i:] = b_scholes(c_current[i, :], strike_price, c_maturity_range[j], risk_free, c_vol[:, j])

    c_fig = go.Figure(data=[go.Surface(z=call_prices, x=c_current[0], y=c_vol[:, 0], colorscale='aggrnyl')])

    c_fig.update_layout(title='3D Surface of Call Option Prices',
                      scene=dict(xaxis_title='Current Price',
                                 yaxis_title='Volatiltiy',
                                 zaxis_title='Call Option price'),
                                 height=600)
    
    st.plotly_chart(c_fig)

with c3:
    
    p_current_range = np.linspace(current_price*0.5, current_price*1.5, 50)
    p_maturity_range = np.linspace(0, 365, 50)
    p_sigma_range = np.linspace(sigma*0.5, sigma*2, 50)

    p_current, p_vol = np.meshgrid(p_current_range, p_sigma_range)

    put_prices = np.zeros(p_current.shape)
    for j in range(p_sigma_range.shape[0]):
        for i in range(p_current.shape[0]):
            put_prices[i:] = b_scholes(p_current[i, :], strike_price, p_maturity_range[j], risk_free, p_vol[:, j], type='put')

    p_fig = go.Figure(data=[go.Surface(z=put_prices, x=p_current[0], y=p_vol[:, 0], colorscale='sunsetdark')])

    p_fig.update_layout(title='3D Surface of Put Option Prices',
                      scene=dict(xaxis_title='Current Price',
                                 yaxis_title='Volatiltiy',
                                 zaxis_title='Put Option price'),
                                 height=600)
    
    st.plotly_chart(p_fig)

with c2:
    st.subheader("Simulated Call Option Greeks")
    c_delta, c_gamma, c_vega, c_theta = calc_greeks(current_price, strike_price, maturity, risk_free, sigma)
    st.markdown(
            f"""
            <div style="padding: 8px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 5px; text-align: center;background-color: #B0E0E6;">
                <strong style="margin-right: 10px;">Delta:</strong> <span style="font-weight: normal;">{c_delta.round(2)}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown(
        f"""
        <div style="padding: 8px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 5px; text-align: center; background-color: #4682B4;">
            <strong style="margin-right: 10px;">Gamma:</strong> <span style="font-weight: normal;">{c_gamma.round(2)}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="padding: 8px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 5px; text-align: center; background-color: #4169E1;">
            <strong style="margin-right: 10px;">Vega:</strong> <span style="font-weight: normal;">{c_vega.round(2)}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="padding: 8px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 5px; text-align: center; background-color: #001f3f;">
            <strong style="margin-right: 10px;">Theta:</strong> <span style="font-weight: normal;">{c_theta.round(2)}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.subheader("Simulated Put Option Greeks")
    p_delta, p_gamma, p_vega, p_theta = calc_greeks(current_price, strike_price, maturity, risk_free, sigma, type='put')
    st.markdown(
            f"""
            <div style="padding: 8px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 5px; text-align: center;background-color: #FFB03B;">
                <strong style="margin-right: 10px;">Delta:</strong> <span style="font-weight: normal;">{p_delta.round(2)}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown(
        f"""
        <div style="padding: 8px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 5px; text-align: center; background-color: #FF6F20;">
            <strong style="margin-right: 10px;">Gamma:</strong> <span style="font-weight: normal;">{p_gamma.round(2)}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="padding: 8px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 5px; text-align: center; background-color: #FF3B00;">
            <strong style="margin-right: 10px;">Vega:</strong> <span style="font-weight: normal;">{p_vega.round(2)}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="padding: 8px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 5px; text-align: center; background-color: #A02800;">
            <strong style="margin-right: 10px;">Theta:</strong> <span style="font-weight: normal;">{p_theta.round(2)}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

