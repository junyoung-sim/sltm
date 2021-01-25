
#from lib.algo import backtest_evaluation
#backtest_evaluation("spy")

import matplotlib.pyplot as plt
from lib import YahooFinance

data = YahooFinance("spy", "2000-01-01", "yyyy-mm-dd").get("prices")
plt.plot(data[-171:])
plt.show()
