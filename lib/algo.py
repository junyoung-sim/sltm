
import os
import numpy as np
import tqdm as tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader

def normalize(data:list):
    return [(p - min(data)) / (max(data) - min(data)) for p in data]

def mavg(data:list, window:int):
    return [sum(data[i:i+window]) / window for i in range(len(data) - window + 1)]

def mse(actual:list, prediction:list):
    return sum([(actual[i] - prediction[i])**2 for i in range(len(actual))]) / len(actual)

def HistoricalData(symbol:str, start:str, end=datetime.today().strftime("%Y-%m-%d")):
    with open("./apikey", "r") as f:
        key = f.readline()
    data = DataReader(symbol, "av-daily-adjusted", start, end, api_key=key)
    # adjusted close price and dates
    data.to_csv("./data/{}.csv" .format(symbol))
    price = list(data["close"]) # adjusted close price
    with open("./data/{}.csv" .format(symbol), "r") as f:
         dates = [line[:10] for line in f.readlines()[1:]] # get date of each price (ignore first line; column description)
    return {"price": price, "dates": dates}

def sample_timeseries_dataset(symbol:str, start:str, end=datetime.today().strftime("%Y-%m-%d")):
    training_input, training_output = [], []
    raw = HistoricalData(symbol, start, end)
    price, dates = raw["price"], raw["dates"]
    # sample time series dataset
    loop = tqdm.tqdm(total=len(price)-204, position=0, leave=False)
    for i in range(len(price)-204):
        loop.set_description("Processing time series... [{} ~ {}]" .format(dates[i], dates[i+204]))
        training_input.append(normalize(mavg(price[i:i+170], 50))) # Normalized D-121 MAVG 50 input
        training_output.append(normalize(mavg(price[i+121:i+205], 10))) # Normalized D+75 MAVG 10 output
        loop.update(1)
    training_input, training_output = np.array(training_input), np.array(training_output)
    with open("./temp/input", "w+") as f:
        for i in range(training_input.shape[0]): # write each input into each line
            for val in training_input[i]:
                f.write("{} " .format(val))
            if i != training_input.shape[0] - 1:
                f.write("\n")
    return {"input": training_input, "output": training_output}

def sample_recent_input(symbol:str):
    data = normalize(mavg(HistoricalData(symbol, "2020-01-01")["price"][-170:], 50))
    with open("./temp/input", "w+") as f:
        for val in data:
            f.write("{} " .format(val))

def validation(model:str):
    model_path = "./models/" + model
    os.system("rm {}/res/validation/*.png" .format(model_path))
    # validate predictions with actual trend data
    raw = HistoricalData(model, "2020-01-01")
    trend, dates = mavg(raw["price"], 10), raw["dates"][10:]
    for f in os.listdir(model_path + "/res/npy"):
        if f.endswith(".npy") and f[:-4] != datetime.today().strftime("%Y-%m-%d"):
            date = f[:-4]
            actual = trend[dates.index(date):]
            if len(actual) >= 2:
                if len(actual) >= 75: # expired predictions
                    actual = normalize(actual[:75])
                    prediction = np.load("{}/res/npy/{}" .format(model_path, f))
                    fig = plt.figure()
                    plt.plot(prediction, color="red")
                    plt.plot(actual, color="green")
                    plt.savefig("{}/res/expired/{}.png" .format(model_path, date))
                    os.system("mv {}/res/npy/{}.npy {}/res/expired/{}.npy" .format(model_path, date, model_path, date))
                    os.system("rm {}/res/prediction/{}.png" .format(model_path, date))
                else:
                    actual = normalize(actual)
                    prediction = normalize(np.load("{}/res/npy/{}" .format(model_path, f))[:len(actual)])
                    error = round(mse(actual, prediction), 4)
                    if error < 0.1:
                        fig = plt.figure()
                        plt.plot(prediction, color="red")
                        plt.plot(actual, color="green")
                        plt.savefig("{}/res/validation/{} [D+{} MSE={}].png" .format(model_path, date, len(actual), error))

