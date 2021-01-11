
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
        if coefficient[i] / abs(coefficient[i]) != coefficient[i+1] / abs(coefficient[i+1]):
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

def backtest_accuracy(model=""):
    actual = list(np.load("./models/" + model + "/backtest/actual.npy"))
    backtest = list(np.load("./models/" + model + "/backtest/output.npy"))
    acc = []
    with open("./models/" + model + "/backtest/accuracy", "w+") as f:
        for d in range(len(actual)):
            actual_direction = [1 if val > 0.00 else 0 for val in differential(actual[d])]
            backtest_direction = [1 if val > 0.00 else 0 for val in differential(backtest[d])]
            correct = sum([1 for i in range(len(actual_direction)) if actual_direction[i] == backtest_direction[i]])
            acc.append(correct * 100 / len(actual_direction))
            print("#{}: {}%" .format(d, acc[-1]))
            f.write("#{}: {}%\n" .format(d, acc[-1]))
    print(sum([1 for val in acc if val > 50.00]) * 100 / len(acc))

def historical_validation(date="yyyy-mm-dd"):
    ids = [i[:i.index("_")] for i in os.listdir("./res/" + date + "/") if i.endswith("_pred.npy")]
    for i in range(len(ids)):
        test_input = list(np.load("./res/" + date + "/" + ids[i] + "_input.npy"))
        raw = YahooFinance(ids[i], "2000-01-01", "yyyy-mm-dd")
        stock, dates = raw.get("prices"), raw.get("dates")
        print("Running historical trend validation... [{}-{}]" .format(date, ids[i]))
        with open("./res/" + date + "/" + ids[i] + "_historical_validation", "w+") as f:
            for i in range(len(stock) - 196):
                historical_input = normalize(mavg(stock[i:i+171], 50))
                historical_differential = [1 if val > 0.00 else 0 for val in differential(historical_input)]
                test_input_differential = [1 if val > 0.00 else 0 for val in differential(test_input)]
                correctness = sum([1 for i in range(len(historical_differential)) if historical_differential[i] == test_input_differential[i]]) / len(test_input_differential)
                if correctness > 0.80:
                    f.write("[{}~{}], [{}~{}] >> {}%\n" .format(dates[i], dates[i+121], dates[i+121], dates[i+196], correctness*100))

def realtime_validation():
    if datetime.datetime.today().weekday() >= 5:
        return
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

