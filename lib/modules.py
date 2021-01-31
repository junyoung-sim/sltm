
import numpy as np
import tqdm as tqdm
from pandas_datareader.data import DataReader

from .algo import *

def YahooFinance(symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd"):
    """ Downloads historical data from Yahoo.
        arg<start>  must be a specific date  """
    if end != "yyyy-mm-dd":
        download = DataReader(symbol, "yahoo", start, end)
    else:
        download = DataReader(symbol, "yahoo", start)
    download.to_csv("./data/" + symbol + ".csv")
    with open("./data/" + symbol + ".csv", "r") as f:
        lines = f.readlines()
    data = list(download["Adj Close"]) # adjusted close price
    dates = [line[:10] for line in lines][1:]
    return {"prices": data, "dates": dates}

def process_timeseries(symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd", write_data=False):
    """ Processes financial time series. ***HARD-CODED PARAMETERS*** """
    input_set, output_set = [], []
    raw = YahooFinance(symbol, start, end)
    stock, dates = raw["prices"], raw["dates"]
    loop = tqdm.tqdm(total=len(stock)-206, position=0, leave=False)          # *** HARD-CODED PARAMETER ***
    for i in range(len(stock)-206):                                          # *** HARD-CODED PARAMETER ***
        loop.set_description("Processing time series... [{}]" .format(dates[i]))
        input_set.append(normalize(mavg(stock[i:i+171], 50)).reshape(11,11)) # *** HARD-CODED PARAMETER ***
        output_set.append(normalize(mavg(stock[i+121:i+206], 10)))           # *** HARD-CODED PARAMETER ***
        loop.update(1)
    input_set, output_set = np.array(input_set), np.array(output_set)
    # write dataset into a file (required when training a model)
    if write_data:
        with open("./temp/input", "w+") as f:
            for i in range(input_set.shape[0]): # very dumb and memory-inefficient (but easy for c code to read)
                for val in input_set[i].flatten():
                    f.write(str(val) + " ")
                if i != input_set.shape[0] - 1:
                    f.write("\n")
    return {"input": input_set, "output": output_set}

