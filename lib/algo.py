
import os
import numpy as np
import tqdm as tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader

def normalize(data=[]):
    return [(p - min(data)) / (max(data) - min(data)) for p in data]

def mavg(data=[], window=int()):
    return [sum(data[i:i+window]) / window for i in range(0, len(data) - window)]

def YahooFinance(symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd"):
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
    input_set, output_set = [], []
    raw = YahooFinance(symbol, start, end)
    stock, dates = raw["prices"], raw["dates"]
    loop = tqdm.tqdm(total=len(stock)-206, position=0, leave=False) # *** HARD-CODED PARAMETER ***
    for i in range(len(stock)-206):                                 # *** HARD-CODED PARAMETER ***
        loop.set_description("Processing time series... [{}]" .format(dates[i]))
        input_set.append(normalize(mavg(stock[i:i+171], 50)))       # *** HARD-CODED PARAMETER *** >> RESHAPE INPUT TO 11x11 (Raster)
        output_set.append(normalize(mavg(stock[i+121:i+206], 10)))  # *** HARD-CODED PARAMETER ***
        loop.update(1)
    input_set, output_set = np.array(input_set), np.array(output_set)
    # write dataset into a file (required when training a model)
    if write_data:
        with open("./temp/input", "w+") as f:
            for i in range(input_set.shape[0]):
                for val in input_set[i]: # write each input in a single line (easy for C code to read)
                    f.write(str(val) + " ")
                if i != input_set.shape[0] - 1:
                    f.write("\n")
    return {"input": input_set, "output": output_set}

def validate_trend_models(model_path="", symbol=""):
    path = model_path + "/res/"
    for f in os.listdir(path):
        if f.endswith(".npy") and f[:-4] != datetime.today().strftime("%Y-%m-%d"):
            sample_date = f[:-4]
            actual = normalize(YahooFinance(symbol, sample_date, datetime.today().strftime("%Y-%m-%d"))["prices"])
            sample = np.load(path + f)[:len(actual)]
            fig = plt.figure()
            plt.plot(sample, color="red")
            plt.plot(actual, color="green")
            plt.savefig(path + sample_date + "-validation.png")
