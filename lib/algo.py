
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

def mse(actual=[], prediction=[]):
    return sum([(actual[i] - prediction[i])**2 for i in range(len(actual))]) / len(actual)

def YahooFinance(symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd"):
    if end != "yyyy-mm-dd":
        download = DataReader(symbol, "yahoo", start, end)
    else:
        download = DataReader(symbol, "yahoo", start)
    download.to_csv("./data/" + symbol + ".csv")
    data = list(download["Adj Close"]) # adjusted close price
    with open("./data/" + symbol + ".csv", "r") as f:
         dates = [line[:10] for line in f.readlines()] # get date of each price
         dates.pop(0)
    return {"prices": data, "dates": dates}

def generate_timeseries_dataset(symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd"):
    training_input, training_output = [], []
    raw = YahooFinance(symbol, start, end)
    stock, dates = raw["prices"], raw["dates"]
    loop = tqdm.tqdm(total=len(stock)-206, position=0, leave=False)
    for i in range(len(stock)-206):
        loop.set_description("Processing time series... [{}]" .format(dates[i]))
        training_input.append(normalize(mavg(stock[i:i+171], 50))) # D-121 MAVG 50 input
        training_output.append(normalize(mavg(stock[i+121:i+206], 10))) # D+75 MAVG 10 output
        loop.update(1)
    training_input, training_output = np.array(training_input), np.array(training_output)
    with open("./temp/input", "w+") as f:
        for i in range(training_input.shape[0]): # write each input into each line
            for val in training_input[i]:
                f.write(str(val) + " ")
            if i != training_input.shape[0] - 1:
                f.write("\n")
    return {"input": training_input, "output": training_output}

def confidence_evaluation(model_path="", symbol=""):
    actual = np.load(model_path + "/backtest/actual.npy")
    backtest = np.load(model_path + "/backtest/backtest.npy")
    error_constant = mse(actual.flatten(), backtest.flatten()) # model cost on entire backtest samples
    ideal_models = [i for i in range(actual.shape[0]) if mse(actual[i], backtest[i]) < error_constant]

    for f in os.listdir(model_path + "/res/npy/"):
        if f.endswith(".npy") and f[:-4] != datetime.today().strftime("%Y-%m-%d"):
            date = f[:-4]
            realtime = normalize(YahooFinance(symbol, date)["prices"])
            if len(realtime) >= 5:
                prediction = np.load(model_path + "/res/npy/" + f)[:len(realtime)]
                error = mse(realtime, prediction)
                # analyze backtest mse distribution on samples with mse lower than error constant
                # and indicate where the evaluating trend model is at
                interval_error = [mse(actual[i][:len(realtime)], backtest[i][:len(realtime)]) for i in ideal_models]
                interval = max(interval_error) / 10
                print("\nTREND MODEL #{}" .format(date))
                print("Confidence Evaluation via Backtest MSE Distribution Analysis [D+0 ~ D+{}]" .format(len(realtime)))
                for i in range(1, 11):
                    print("{}: " .format(round(interval * i, 5)), end="")
                    distribution = 0
                    for val in interval_error:
                        if val > interval * (i - 1) and val < interval * i:
                            print("*", end="")
                            distribution += 1
                    print(" ({}%)" .format(round(distribution * 100 / len(interval_error), 5)), end="")
                    if error > interval * (i - 1) and error < interval * i:
                        print("            <======== PREDICTED MODEL", end="")
                    print("")

#def trend_validation(model_path="", symbol=""):
#    path = model_path + "/res/npy/"
#    raw = YahooFinance(symbol, "2021-01-01")
#    stock, dates = mavg(raw["prices"], 10), raw["dates"][10:]
#    print("\n|[ Trend Validation Results #]|")
#    print("------------ Date ------------ Direction Accuracy ------------ MSE ------------")
#    for f in os.listdir(path):
#        if f.endswith(".npy"):
#            date = f[:-4]
#            if date != datetime.today().strftime("%Y-%m-%d"):
#                actual = normalize(stock[dates.index(date):])
#                prediction = np.load(path + f)[:len(actual)]
#                actual_derivative = [actual[i+1] - actual[i] for i in range(len(actual) - 1)]
#                prediction_derivative = [prediction[i+1] - prediction[i] for i in range(len(prediction) - 1)]
#                accuracy = int()
#                for i in range(len(actual_derivative)):
#                    if abs(actual_derivative[i]) / actual_derivative[i] == abs(prediction_derivative[i]) / prediction_derivative[i]:
#                        accuracy += 1
#                accuracy *= int(100 / len(actual_derivative))
#                mse = sum([(actual[i] - prediction[i])**2 for i in range(len(actual))]) / len(actual)
#                print("          {}                   {}%               {}" .format(date, accuracy, mse))

