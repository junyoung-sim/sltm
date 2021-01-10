
import tqdm as tqdm
import numpy as np
import os, ast, json
import matplotlib.pyplot as plt

from .algo import normalize
from .util import *
from .dnn import DeepNeuralNetwork

np.set_printoptions(suppress=True)

def relu(value=float(), slope=1.00):
    return max(0, value * slope)

def extract_features(data=[], kernel=[], stride=1, padding=True): # convolution
    if padding is True:
        pad = np.zeros((data.shape[0] + 2, data.shape[1] + 2))
        for i in range(1, data.shape[0] - 1):
            for j in range(1, data.shape[1] - 1):
                pad[i][j] = data[i-1][j-1]
        data = pad
    encoded = []
    for r in range(0, data.shape[0] - kernel.shape[0] + 1, stride):
        encoded_row = []
        for c in range(0, data.shape[1] - kernel.shape[1] + 1, stride):
            matmul = 0.00
            k_row, k_col = 0, 0
            for i in range(r, r + kernel.shape[0]):
                for j in range(c, c + kernel.shape[1]):
                    matmul += data[i][j] * kernel[k_row][k_col]
                    k_col += 1
                k_row += 1
                k_col = 0
            encoded_row.append(relu(matmul, 1.0))
        encoded.append(encoded_row)
    encoded = np.array(encoded)
    return encoded
 
def size_reduction(data=[], pool_type="", pool_size=[2,2]): # pooling (max/average)
    reduced = []
    for r in range(0, data.shape[0] - pool_size[0] + 1, pool_size[0]):
        reduced_row = []
        for c in range(0, data.shape[1] - pool_size[1] + 1, pool_size[1]):
            window = [data[i][j] for j in range(c, c + pool_size[1]) for i in range(r, r + pool_size[0])]
            if pool_type == "max":
                reduced_row.append(max(window))
            elif pool_type == "avg":
                reduced_row.append(sum(window) / len(window))
        reduced.append(reduced_row)
    reduced = np.array(reduced)
    return reduced

class Model:
    def __init__(self, path=""):
        self.path = path
        self.input, self.output = [], []
        self.layer, self.kernel = [], []
    def save(self):
        with open(self.path + "encoder_attributes", "w+") as File:
            File.write(json.dumps(self.layer))
        for i in range(self.kernel.shape[0]):
            write(self.kernel[i], self.path + "kernel" + str(i))
    def load(self):
        try:
            with open(self.path + "encoder_attributes", "r") as File:
                self.layer = json.loads(File.read())
            for i in range(len(self.layer)):
                self.kernel.append(np.array(read(self.path + "kernel" + str(i))))
            self.kernel = np.array(self.kernel)
            return True
        except FileNotFoundError:
            return False
    def initialize(self, dataset={"input":[], "output":[]}):
        self.input, self.output = dataset.get("input"), dataset.get("output")
        if self.load() != True:
            for layer in self.layer:
                self.kernel.append(np.random.uniform(-0.5, 0.5, size=(layer.get("conv_size")[0], layer.get("conv_size")[1])))
            self.kernel = np.array(self.kernel)
        for i in range(len(self.layer)):
            print("Encoding Layer #{} = {}" .format(i, self.layer[i]))
    def add_layer(self, attributes={"conv_size":[2,2], "stride":1, "padding":True, "pool_type":"max", "pool_size":[2,2]}):
        self.layer.append(attributes) # encoder_attributes
    def encode(self):
        encoded = []
        loop = tqdm.tqdm(total=self.input.shape[0], position=0, leave=False)
        for data in self.input: # sequentially encode each data through the encoding layers
            loop.set_description("Encoding dataset...")
            for l in range(len(self.layer)):
                stride, padding = self.layer[l].get("stride"), self.layer[l].get("padding")
                pool_type, pool_size = self.layer[l].get("pool_type"), self.layer[l].get("pool_size")
                data = extract_features(data, self.kernel[l], stride, padding)
                data = size_reduction(data, pool_type, pool_size)
            encoded.append(data)
            loop.update(1)
        self.input = np.array(encoded) 
        print("\n", self.input)
    def train(self, learning_rate=0.01, iteration=10000, test=0):
        self.encode()
        encoded = np.array([data.flatten() for data in self.input])
        output = np.array([normalize(data) for data in self.output])
        decoder_config = {
            "layer_config": [[25,25],[25,100],[100,75]], # *** PARAMETER TUNING ***
            "activation": "relu",
            "abs_synapse": 1.00,
            "cost": "MSE",
            "learning_rate": learning_rate
        }
        decoder = DeepNeuralNetwork(self.path, decoder_config)
        # testing dataset will automatically partitioning before the optimization
        test_result = decoder.train(encoded, output, iteration, 0.05 if test != 0 else 0.00)
        self.save()
        return test_result
    def run(self):
        self.encode()
        encoded = np.array([data.flatten() for data in self.input])
        decoder = DeepNeuralNetwork(self.path)
        constructed_output = decoder.predict(encoded)
        return constructed_output

