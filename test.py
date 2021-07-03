import yfinance as yf
import pandas_datareader.data as pdr

yf.pdr_override()

print(pdr.get_data_yahoo("SPY", "2000-01-01"))
