
import numpy as np

def normalize(data=[]):
    return np.array([(p - min(data)) / (max(data) - min(data)) for p in data])

def mavg(data=[], window=int()):
    return np.array([sum(data[i:i+window]) / window for i in range(0, len(data) - window)])
