
from lib import HistoricalData
import matplotlib.pyplot as plt

data = HistoricalData("spy", "2010-01-01")["price"]
plt.plot(data, color="red")
plt.show()
