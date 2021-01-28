
import os
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt
import datetime

from .algo import *
from .model import *

def plot_results(results=[], actual=[], save_path=""): # use for comparing corresponding plots
    if os.path.exists(save_path) != True:
        os.mkdir(save_path)
    loop = tqdm.tqdm(total=results.shape[0], position=0, leave=False)
    for i in range(results.shape[0]):
        loop.set_description("Saving samples...")
        fig = plt.figure()
        plt.plot(results[i], color="red")
        plt.plot(actual[i], color="green")
        plt.savefig(save_path + "sample" + str(i) + ".png")
        loop.update(1)

def process_timeseries(symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd"): # generate dataset
    input_set, output_set = [], []
    stock = YahooFinance(symbol.lower(), start, end)["prices"]
    loop = tqdm.tqdm(total=len(stock)-206, position=0, leave=False)
    for i in range(0, len(stock)-206):
        loop.set_description("Processing time series...")
        input_set.append(normalize(mavg(stock[i:i+171], 50)).reshape(11,11)) # D-121 Long Term Trend  (Input) *** PARAMETER TUNING ***
        output_set.append(normalize(mavg(stock[i+121:i+206], 10)))           # D+75 Short Term Trend (Output) *** PARAMETER TUNING ***
        loop.update(1)
    return {"input": np.array(input_set), "output": np.array(output_set)}

class Futures:
    def __init__(self, name="", portfolio=""):
        self.name = name
        self.model_path = "./models/" + name + "/"
        if os.path.exists(self.model_path) != True:
            os.mkdir(self.model_path)
        self.model = Model(path=self.model_path) # deep neural network model used for decoding
    def train(self, symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd", learning_rate=0.01, iteration=10000, test=0):
        dataset = process_timeseries(symbol, start, end)
        self.model.add_layer({"conv_size":[2,2], "stride":1, "padding":False, "pool_type":"max", "pool_size":[2,2]})
        self.model.initialize(dataset)
        test_result = self.model.train(learning_rate, iteration, test)
        if test != 0:
            actual = dataset.get("output")[-test_result.shape[0]:]
            plot_results(test_result, actual, self.model_path + "backtest/") 
            with open(self.model_path + "backtest/actual.npy", "wb") as f:
                np.save(f, actual)
            with open(self.model_path + "backtest/output.npy", "wb") as f:
                np.save(f, test_result)
            backtest_evaluation(self.name)
    def save_trained_data(self, symbol="", start="yyyy-mm-dd", end="yyyy-mm-dd"):
        dataset = process_timeseries(symbol, start, end)
        self.model.initialize(dataset)
        prediction = self.model.run()
        plot_results(prediction, dataset.get("output"), self.model_path + "trained-samples/")
    def run(self, symbol=""): # get predictions from current time period
        stock = YahooFinance(symbol.lower(), "2015-01-01").get("prices")
        test_input = np.array([normalize(mavg(stock[-171:], 50)).reshape(11,11)]) # *** PARAMETER TUNING *** 
        self.model.initialize({"input": test_input, "output": []})
        prediction = self.model.run()

        save_path = "./res/" + str(datetime.date.today().strftime("%Y-%m-%d")) + "/"
        if os.path.exists(save_path) != True:
            os.mkdir(save_path)
        with open(save_path + symbol.lower() + "_input.npy", "wb") as f:
            fig = plt.figure()
            plt.plot(test_input[0].flatten(), color="green")
            plt.savefig(save_path + symbol.lower() + "_input.png")
            np.save(f, test_input[0].flatten())
        with open(save_path + symbol.lower() + "_pred.npy", "wb") as f:
            fig = plt.figure()
            plt.plot(prediction[0], color="red")
            plt.savefig(save_path + symbol.lower() + "_pred.png")
            np.save(f, prediction[0])
        with open(save_path + self.name, "w+") as summary:
            print(symbol + ":")
            summary.write(symbol + ":\n")
            for instruction in differential_analysis(prediction[0]): # identify extremum points from prediction
                print("  " + instruction)
                summary.write("  " + instruction + "\n")
        realtime_validation()

