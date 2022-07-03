import select
from matplotlib.axis import XAxis
import streamlit as st 
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Prediksi Harga Saham ")

stocks = ("AAPL", "MSFT", "AMZN", "FB", "GOOG", "INTC", "CSCO", "IBM", "ORCL", "QCOM", "TXN", "XOM")
selected_stock = st.selectbox("Pilih Saham", stocks)

n_years = st.slider("jumlah tahun prediksi", 1, 10, 1)
period = n_years *365


# fungsi untuk memuat data dari yahoo finance
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start=START, end=TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Memuat data...")
data = load_data(selected_stock)
data_load_state.text("Data telah dimuat")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Open'))
    fig.layout.update(title_text="Harga Saham Terkini", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()
    
#forecasting
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Prediksi Harga Saham')
st.write(forecast.tail())

st.write('Prediksi harga saham untuk ' + str(n_years) + ' tahun kedepan')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Prediksi detail harga saham untuk ' + str(n_years) + ' tahun kedepan')
fig2 = m.plot_components(forecast)
st.write(fig2)