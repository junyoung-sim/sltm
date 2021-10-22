#!/usr/bin/env python3

import sys
import tqdm as tqdm
from datetime import datetime
from pandas_datareader.data import DataReader

def normalize(data:list):
    normalized = []
    try:
        normalized = [(p - min(data)) / (max(data) - min(data)) for p in data]
    except Exception as e:
        normalized = data
    return normalized

def mse(y:list, yhat:list):
    return sum([(y[i] - yhat[i])**2 for i in range(len(y))]) / len(y)

def mavg(data:list, window:int):
    return [sum(data[i:i+window]) / window for i in range(len(data) - window + 1)]

def HistoricalData(symbol:str, start:str, end=datetime.today().strftime("%Y-%m-%d")):
    with open("./apikey", "r") as f:
        key = f.readline()
    data = DataReader(symbol, "av-daily-adjusted", start, end, api_key=key)
    # adjusted close price and dates
    data.to_csv("./data/{}.csv" .format(symbol))
    price = list(data["adjusted close"]) # adjusted close price
    with open("./data/{}.csv" .format(symbol), "r") as f:
         dates = [line[:10] for line in f.readlines()[1:]] # get date of each price (ignore first line; column description)
    return {"price": price, "dates": dates}

def sample_timeseries_dataset(symbol:str, start:str, end=datetime.today().strftime("%Y-%m-%d")):
    input_vectors, output_vectors = [], []
    raw = HistoricalData(symbol, start, end)
    price, dates = raw["price"], raw["dates"]
    # sample time series dataset
    loop = tqdm.tqdm(total=len(price)-180, position=0, leave=False)
    for i in range(len(price)-180):
        loop.set_description("Processing time series... [{} ~ {}]" .format(dates[i], dates[i+180]))
        input_vectors.append(normalize(mavg(price[i:i+170], 50))) # Normalized D-121 MAVG 50 input
        output_vectors.append(normalize(mavg(price[i+121:i+180], 10))) # Normalized D+50 MAVG 10 output
        loop.update(1)
    # write dataset to ./temp
    with open("./temp/input", "w+") as f1, open("./temp/output", "w+") as f2:
        for i in range(len(input_vectors)):
            for val in input_vectors[i]:
                f1.write("{} " .format(val))
            for val in output_vectors[i]:
                f2.write("{} " .format(val))
            if i != len(input_vectors) - 1:
                f1.write("\n")
                f2.write("\n")
    return input_vectors, output_vectors

def sample_recent_input(symbol:str):
    data = normalize(mavg(HistoricalData(symbol, "2020-01-01")["price"][-170:], 50))
    with open("./temp/input", "w+") as f:
        for val in data:
            f.write("{} " .format(val))

def main():
    if sys.argv[1] == "sample_timeseries_dataset":
        sample_timeseries_dataset(symbol=sys.argv[2], start=sys.argv[3], end=sys.argv[4])
    elif sys.argv[1] == "sample_recent_input":
        sample_recent_input(symbol=sys.argv[2])
    else:
        pass   

if __name__ == "__main__":
    main()
