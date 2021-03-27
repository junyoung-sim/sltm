
import os
import numpy as np
import tqdm as tqdm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader

def normalize(data=[]):
    return [(p - min(data)) / (max(data) - min(data)) for p in data]

def mavg(data=[], window=int()):
    return [sum(data[i:i+window]) / window for i in range(0, len(data) - window)]

def smoothing(line=[]):
    smooth = []
    kernel = [-3.0, 12.0, 17.0, 12.0, -3.0] # Savitzky-Golay filter
    for p in range(len(line) - len(kernel) + 1):
        k_i, matmul = 0, 0.00
        for i in range(p, p + len(kernel)):
            matmul += line[i] * kernel[k_i]
            k_i += 1
        k_i = 0
        smooth.append(matmul/35) # 35 is normalization constant
    return smooth

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

def generate_timeseries_dataset(symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd"):
    input_set, output_set = [], []
    raw = YahooFinance(symbol, start, end)
    stock, dates = raw["prices"], raw["dates"]
    loop = tqdm.tqdm(total=len(stock)-206, position=0, leave=False) # *** HARD-CODED PARAMETER ***
    for i in range(len(stock)-206):                                 # *** HARD-CODED PARAMETER ***
        loop.set_description("Processing time series... [{}]" .format(dates[i]))
        input_set.append(normalize(mavg(stock[i:i+171], 50)))       # *** HARD-CODED PARAMETER ***
        output_set.append(normalize(mavg(stock[i+121:i+206], 10)))  # *** HARD-CODED PARAMETER ***
        loop.update(1)
    input_set, output_set = np.array(input_set), np.array(output_set) 
    with open("./temp/input", "w+") as f:
        for i in range(input_set.shape[0]):
            for val in input_set[i]: # write each input in a single line (easy for C code to read)
                f.write(str(val) + " ")
            if i != input_set.shape[0] - 1:
                f.write("\n")
    return {"input": input_set, "output": output_set}

def trend_validation(model_path="", symbol=""):
    path = model_path + "/res/npy/"
    raw = YahooFinance(symbol, "2021-01-01")
    stock = mavg(raw["prices"], 10)
    dates = raw["dates"][10:]
    print("\n|[ Trend Validation Results ]|")
    print("------------ Date ------------ Direction Accuracy ------------ MSE ------------")
    # validate each trend model saved in model
    for f in os.listdir(path):
        if f.endswith(".npy"):
            date = f[:-4]
            if date != datetime.today().strftime("%Y-%m-%d") and date != "2021-03-26":
                actual = normalize(stock[dates.index(date):])
                prediction = np.load(path + f)[:len(actual)]
                # calculate directional accuracy
                actual_derivative = [actual[i+1] - actual[i] for i in range(len(actual) - 1)]
                prediction_derivative = [prediction[i+1] - prediction[i] for i in range(len(prediction) - 1)]
                accuracy = sum([1 for i in range(len(actual_derivative)) if abs(actual_derivative[i]) / actual_derivative[i] == abs(prediction_derivative[i]) / prediction_derivative[i]]) * 100 / len(actual_derivative)
                # calculate MSE
                mse = sum([(actual[i] - prediction[i])**2 for i in range(len(actual))]) / len(actual)
                # show validation result
                print("          {}                   {}%          {}" .format(date, int(accuracy), mse))
                plt.plot(actual, color="green")
                plt.plot(prediction, color="red")
                plt.show()
