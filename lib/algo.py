
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

def realtime_mse(model_path="", symbol=""):
    raw = YahooFinance(symbol, "2021-01-01")
    mavg10, dates = mavg(raw["prices"], 10), raw["dates"][10:]
    # calculate realtime mse of prediction models that are 5 days old or more
    evaluating_predictions = []
    for f in os.listdir(model_path + "/res/npy/"):
        if f.endswith(".npy") and f[:-4] != datetime.today().strftime("%Y-%m-%d"):
            date = f[:-4]
            realtime = normalize(mavg10[dates.index(date):])
            if len(realtime) >= 5:
                print("{} [D+{}]: " .format(date, len(realtime)), end="")
                prediction = normalize(np.load(model_path + "/res/npy/" + f)[:len(realtime)])
                error = mse(realtime, prediction)
                print("MSE = {}" .format(round(error, 5)))
                evaluating_predictions.append({"date": date, "realtime": realtime, "prediction": prediction, "error": error})
    return evaluating_predictions

def confidence_evaluation(model_path="", evaluating_predictions=[]):
    lowest_mse = min([model["error"] for model in evaluating_predictions])
    # find trend model with lowest mse (best model)
    for model in evaluating_predictions:
        if model["error"] == lowest_mse:
            best = model
            print("\nBEST MODEL = {}" .format(best["date"]))
    # Confidence Evaluation via Backtest MSE Distribution Analysis
    # find backtest samples with a mse lower than entire model's cost (ideal backtest samples)
    actual = np.load(model_path + "/backtest/actual.npy")
    backtest = np.load(model_path + "/backtest/backtest.npy")
    general_cost = mse(actual.flatten(), backtest.flatten())
    ideal = [i for i in range(actual.shape[0]) if mse(actual[i], backtest[i]) < general_cost]
    # calculate the mse of ideal samples within the realtime segment length
    segment_error = [mse(actual[i][:len(best["realtime"])], backtest[i][:len(best["realtime"])]) for i in ideal]
    interval = max(segment_error) / 10
    # detect range of mse with highest distribution probability among the ideal samples (confidence range)
    majority, majority_interval = 0.00, 0.00
    for i in range(1, 11):
        distribution = 0
        for val in segment_error:
            if val > interval * (i - 1) and val < interval * i:
                distribution += 1
        probability = distribution / len(segment_error)
        if probability > majority:
            majority = probability
            majority_interval = interval * i
    # check if the best model is within the confidence range
    print("CONFIDENCE = ", end="")
    if best["error"] > majority_interval - interval and best["error"] < majority_interval:
        print("{}" .format(round(majority, 5)))
    else:
        print("negative")

