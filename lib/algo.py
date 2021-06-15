
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

def mse(actual=[], prediction=[]):
    return sum([(actual[i] - prediction[i])**2 for i in range(len(actual))]) / len(actual)

def vector_analysis(actual=[], prediction=[]):
    error = mse(actual, prediction)
    # directional accuarcy
    actual_derivative = [actual[i+1] - actual[i] for i in range(len(actual) - 1)]
    prediction_derivative = [prediction[i+1] - prediction[i] for i in range(len(prediction) - 1)]
    directional_accuracy = 0
    for i in range(len(actual_derivative)):
        if abs(actual_derivative[i]) / actual_derivative[i] == abs(prediction_derivative[i]) / prediction_derivative[i]:
            directional_accuracy += 1
    directional_accuracy *= 100 / len(actual_derivative)
    return directional_accuracy * (1 - error) # vector score

def RealtimePrice(symbol=""):
    price = list(DataReader(symbol, "yahoo", datetime.today().strftime("%Y-%m-%d"))["Adj Close"])[0]
    return price

def HistoricalData(symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd"):
    if end != "yyyy-mm-dd":
        download = DataReader(symbol, "yahoo", start, end)
    else:
        download = DataReader(symbol, "yahoo", start)
    download.to_csv("./data/{}.csv" .format(symbol))
    price = list(download["Adj Close"]) # adjusted close price
    with open("./data/{}.csv" .format(symbol), "r") as f:
         dates = [line[:10] for line in f.readlines()] # get date of each price
         del dates[0] # first line is category header
    return {"price": price, "dates": dates}

def generate_timeseries_dataset(symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd"):
    training_input, training_output = [], []
    raw = HistoricalData(symbol, start, end)
    price, dates = raw["price"], raw["dates"]
    # sample time series dataset
    loop = tqdm.tqdm(total=len(price)-206, position=0, leave=False)
    for i in range(len(price)-206):
        loop.set_description("Processing time series... [{}]" .format(dates[i]))
        training_input.append(normalize(mavg(price[i:i+171], 50))) # D-121 MAVG 50 input
        training_output.append(normalize(mavg(price[i+121:i+206], 10))) # D+75 MAVG 10 output
        loop.update(1)
    training_input, training_output = np.array(training_input), np.array(training_output)
    with open("./temp/input", "w+") as f:
        for i in range(training_input.shape[0]): # write each input into each line
            for val in training_input[i]:
                f.write(str(val) + " ")
            if i != training_input.shape[0] - 1:
                f.write("\n")
    return {"input": training_input, "output": training_output}

def local_minmax(model="", prediction_date=""):
    raw = HistoricalData(model, "2000-01-01")

def validation(model=""):
    model_path = "./models/" + model
    os.system("rm {}/res/validation/*.png" .format(model_path))
    # validate predictions with real-time data
    raw = HistoricalData(model, "2020-01-01")
    trend, dates = mavg(raw["price"], 10), raw["dates"][10:]
    best = {"vector_score": 0.0, "date": "yyyy-mm-dd"}
    for f in os.listdir(model_path + "/res/npy"):
        if f.endswith(".npy") and f[:-4] != datetime.today().strftime("%Y-%m-%d"):
            date = f[:-4]
            actual = normalize(trend[dates.index(date):])
            if len(actual) > 10 and len(actual) <= 75:
                prediction = normalize(np.load("{}/res/npy/{}" .format(model_path, f))[:len(actual)])
                vector_score = round(vector_analysis(actual, prediction), 2)
                # output and save validation results with vector score higher than 75%
                if vector_score > 70.00:
                    fig = plt.figure()
                    plt.plot(actual, color="green")
                    plt.plot(prediction, color="red")
                    plt.savefig("{}/res/validation/{} [D+{} | VS={}].png" .format(model_path, date, len(actual), vector_score))
                    # identify prediction with highest vector score
                    if vector_score > best["vector_score"]:
                        best["vector_score"] = vector_score
                        best["date"] = date
    print("Best Performing Model:\n    {}-{}: Vector Score = {}" .format(model, best["date"], best["vector_score"]))

