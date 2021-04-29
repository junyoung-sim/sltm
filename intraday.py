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

def refresh(seconds=int()):
    sleep(seconds)
    system("clear")

def run():
    # wait for market open
    market_open = datetime.strptime(str(datetime.now())[:10] + " 09:30:00", "%Y-%m-%d %H:%M:%S")
    while True:
        now = datetime.now()
        if str(market_open) != datetime.strftime(now, "%Y-%m-%d %H:%M:%S"):
            time_left = str(market_open - now)[:-7]
            print("Waiting for market open... ({} left)" .format(time_left))
        else:
            break
        refresh(seconds=1)
    # sample first n minutes of intraday price data
    intraday = []
    delay = 0
    for minute in range(30):
        start = time()
        intraday.append(RealtimePrice("spy"))
        end = time() # record time delay that occurred while reading price data
        print("SPY @{} = ${}" .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), round(intraday[-1], 2)))
        delay = end - start
        sleep(60 - delay)
    # TEMPORARY PROCESS: save intraday data
    with open("./data/{}.itd" .format(datetime.today().strftime("%Y-%m-%d")), "w+") as f:
        for i in range(len(intraday)):
            f.write(intraday[i])
            if i != len(intraday) - 1:
                f.write("\n")

if __name__ == "__main__":
    run()

