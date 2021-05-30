#!/usr/bin/env python3

import sys
from lib import HistoricalData, mavg
import matplotlib.pyplot as plt

symbol = sys.argv[1]
date1  = sys.argv[2]
date2  = sys.argv[3]

data = HistoricalData(symbol, date1, date2)
raw, dates = data["price"][10:], data["dates"][10:]
trend = mavg(data["price"], 10)

direction = [1 if trend[i+1] - trend[i] > 0 else 0 for i in range(len(trend)-1)]
local_min = [i for i in range(len(direction)-1) if direction[i] == 0 and direction[i+1] == 1]
local_max = [i for i in range(len(direction)-1) if direction[i] == 1 and direction[i+1] == 0]

local_min_constant = sum([(min(raw[i-10 if i >= 9 else 0:i]) - trend[i]) * 100 / trend[i] for i in local_min if i != 0]) / len(local_min)
local_max_constant = sum([(max(raw[i-10 if i >= 9 else 0:i]) - trend[i]) * 100 / trend[i] for i in local_max if i != 0]) / len(local_max)

print("Local Minimum Constant = {}%" .format(round(local_min_constant, 2)))
print("Local Maximum Constant = {}%" .format(round(local_max_constant, 2)))
