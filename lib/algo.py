
import os, random, datetime, itertools
import matplotlib.pyplot as plt
import numpy as np

from .util import YahooFinance

def normalize(data):
    return np.array([(p - min(data)) / (max(data) - min(data)) for p in data])

def mavg(timeseries=[], window=50):
    return np.array([sum(timeseries[i:i+window]) / window for i in range(0, len(timeseries) - window)])

def differential(line=[]):
    return [line[i+1] - line[i] for i in range(len(line) - 1)]

def differential_analysis(line=[]):
    coefficient = differential(line)
    curve_point = []
    for i in range(len(coefficient) - 1):
        if coefficient[i] / abs(coefficient[i]) != coefficient[i+1] / abs(coefficient[i+1]): # identify when direction of slope alters
            curve_point.append(i)
    dates, count = [], 0
    while len(dates) != len(line):
        date = datetime.datetime.today() + datetime.timedelta(days=count)
        if date.weekday() < 5: # avoid weekends
            dates.append(date.strftime("%Y-%m-%d"))
        count += 1
    instructions = []
    for index in curve_point:
         instruction = str(index) + " >> " + dates[index] + ": Differential Coefficient = " + str(coefficient[index]) + " "
         if coefficient[index] < 0 and coefficient[index + 1] > 0:
             instruction += "BUY"
         elif coefficient[index] > 0 and coefficient[index + 1] < 0:
             instruction += "SELL"
         else:
             instruction += "HOLD"
         instructions.append(instruction)
    return instructions

def backtest_evaluation(model=""):
    actual = list(np.load("./models/" + model + "/backtest/actual.npy"))
    backtest = list(np.load("./models/" + model + "/backtest/output.npy"))
    accuracy, correctness_probability = [], float()
    with open("./models/" + model + "/backtest/accuracy", "w+") as f:
        for d in range(len(actual)): # evaluate prediction accuracy and correctness probability
            actual_direction = [1 if val > 0.00 else 0 for val in differential(actual[d])]
            backtest_direction = [1 if val > 0.00 else 0 for val in differential(backtest[d])]
            correct = sum([1 for i in range(len(actual_direction)) if actual_direction[i] == backtest_direction[i]])
            accuracy.append(correct * 100 / len(actual_direction))
            f.write("#{}: {}%\n" .format(d, accuracy[-1]))
        correctness_probability = sum([1 for val in accuracy if val > 65.00]) * 100 / len(accuracy)
        f.write("Backtest Correctness Probability: {}%" .format(correctness_probability))
    # evaluate expected profit from each prediction for performance evaluation
    ratio = 0.6
    stock = 0
    shares = 0
    cash = 1000
    balance = cash
    # increase stock asset ratio during an increasing trend
    # decrease stock asset ratio during an decreasing trend
    profit = []
    #for pred in backtest:

def realtime_validation():
    dates = [d for d in os.listdir("./res/") if d != ".DS_Store"]
    for date in dates:
        if date != datetime.datetime.today().strftime("%Y-%m-%d"):
            ids = [i[:i.index("_")] for i in os.listdir("./res/" + date + "/") if i.endswith("_pred.npy")]
            for i in range(len(ids)):
                prediction = list(np.load("./res/" + date + "/" + ids[i] + "_pred.npy"))
                actual = YahooFinance(ids[i], date, "yyyy-mm-dd").get("prices")
                if len(actual) >= 2:
                    fig = plt.figure()
                    plt.plot(prediction[:len(actual)], color="red")
                    plt.plot(normalize(actual), color="green")
                    plt.savefig("./res/" + date + "/" + ids[i] + "_realtime_validation.png")
                    print("Saved real-time validation result of [{}-{}] trend model" .format(date, ids[i]))

