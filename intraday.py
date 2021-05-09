#!/usr/bin/env python3

""" 
Intraday trading model using Futures.
Trained on SPY (S&P 500)
Applicable on SPY, DIA, QQQ (depending on performance)
Run this script during the same day before the next market open.
"""

import sys
from os import system
from time import time, sleep
from datetime import datetime
import matplotlib.pyplot as plt

from lib import *

model  = sys.argv[1] 
symbol = sys.argv[2]

def refresh(seconds=int()):
    sleep(seconds)
    system("clear")

def run():
    # wait for market open
    market_open = datetime.strptime(str(datetime.now())[:10] + " 09:30:00", "%Y-%m-%d %H:%M:%S")
    while True:
        now = datetime.now()
        if str(market_open) != datetime.strftime(now, "%Y-%m-%d %H:%M:%S"):
            time_left = str(market_open - now)[:-7] # omit microseconds
            print("Waiting for market open... ({} left)" .format(time_left))
        else:
            break
        refresh(seconds=1)
    # sample intraday 1 minute price data
    delay = 0
    intraday = []
    market_close = datetime.strptime(str(datetime.now())[:10] + " 16:00:00", "%Y-%m-%d %H:%M:%S")
    while True:
        now = datetime.now()
        if str(market_close) != datetime.strftime(now, "%Y-%m-%d %H:%M:%S"):
            start = time()
            intraday.append(RealtimePrice(symbol))
            end = time()
            delay = end - start
            print("{} @{}: ${}" .format(symbol, datetime.strftime(now, "%Y-%m-%d %H:%M:%S"), intraday[-1]))
            sleep(60 - delay)
        else:
            break
    # save the data of the last 120 minutes in ./temp/input
    with open("./temp/input", "w+") as f:
        for val in normalize(intraday[-121:]):
            f.write(str(val) + " ")
    # run encoder
    system("./encoder " + model)
if __name__ == "__main__":
    run()

