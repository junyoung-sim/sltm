#!/usr/bin/env python3

# Specify these arguments in "order" to execute
#  0     1     2         3        4           5            6         7
# mode model symbol start_date end_date learning_rate, iteration, backtest

from core import Futures

if __name__ == "__main__":
    with open("order", "r") as order:
        command = order.read().split()
    model = Futures(command[1])
    if command[0] == "train":
        model.train(command[2], command[3], command[4], float(command[5]), int(command[6]), int(command[7]))
    elif command[0] == "run":
        model.run(command[2])
    elif command[0] == "test_trained":
        model.test_trained_data(command[2], command[3], command[4])
    else:
        print("Invalid argument was given")

