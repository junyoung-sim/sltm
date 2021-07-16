
import tqdm as tqdm
import os, logging, json, ast
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
from .algo import normalize, mse

# suppress warnings from TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
tf.disable_v2_behavior()

class DeepNeuralNetwork:
    def __init__(self, path:str):
        self.path = path
        # set hyperparameters
        architecture  = [[25,25],[25,100],[100,75]] # [25,25],[25,100],[100,75]
        activation    = "relu"
        abs_synapse   = 1.00
        learning_rate = 0.01
        # setup the neural network (weights, hidden layers, and optimzer)
        with tf.Graph().as_default():
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.input, self.output = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
            self.weights = [tf.Variable(tf.random_uniform([layer[0], layer[1]], -abs_synapse, abs_synapse)) for layer in architecture]
            self.layer = []
            for l in range(len(architecture)):
                if l == 0: # input layer
                    if activation == "relu":
                        self.layer.append(tf.nn.relu(tf.matmul(self.input, self.weights[l])))
                    elif activation == "sigmoid":
                        self.layer.append(tf.nn.sigmoid(tf.matmul(self.input, self.weights[l])))
                    elif activation == "tanh":
                        self.layer.append(tf.nn.tanh(tf.matmul(self.input, self.weights[l])))
                else:
                    if activation == "relu":
                        self.layer.append(tf.nn.relu(tf.matmul(self.layer[l-1], self.weights[l])))
                    elif activation == "sigmoid":
                        self.layer.append(tf.nn.sigmoid(tf.matmul(self.layer[l-1], self.weights[l])))
                    elif activation == "tanh":
                        self.layer.append(tf.nn.tanh(tf.matmul(self.layer[l-1], self.weights[l])))
            # setup cost function and optimizer
            self.cost = tf.reduce_mean(tf.squared_difference(self.output, self.layer[-1])) # MeanSquaredError
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.model = self.optimizer.minimize(self.cost, global_step=self.global_step)
            # setup TensorFlow session and saver
            self.sess = tf.Session()
            self.saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.path + "/dnn")
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.saver.restore(self.sess, ckpt.model_checkpoint_path) # load existing sessions saved in model directory
            else:
                self.sess.run(tf.global_variables_initializer())
    def train(self, dataset={"input": [], "output": []}, iteration:int, test=0.00):
        training_input, training_output = dataset["input"], dataset["output"]
        if test != 0.00:
            # partition dataset
            total_samples = training_input.shape[0]
            test_input = training_input[-int(total_samples * test):]
            test_output = training_output[-int(total_samples * test):]
            training_input = training_input[:-int(total_samples * test)]
            training_output = training_output[:-int(total_samples * test)]
        # training!
        for i in range(iteration):
            self.sess.run(self.model, feed_dict={self.input: training_input, self.output: training_output})
            if i % (iteration / 10) == 0:
                cost = self.sess.run(self.cost, feed_dict={self.input: training_input, self.output: training_output})
                print("ITERATION #{}: COST = {}" .format(i, cost))
        self.saver.save(self.sess, self.path + "/dnn/checkpoint.ckpt", global_step=self.global_step)
        if test != 0.00:
            backtest = np.array([normalize(result) for result in self.sess.run(self.layer[-1], feed_dict={self.input: test_input})])
            print("BACKTEST COST = ", mse(test_output.flatten(), backtest.flatten()))            
            # save backtesting samples
            loop = tqdm.tqdm(total=len(backtest), position=0, leave=False)
            for i in range(len(backtest)):
                loop.set_description("Saving backtested samples... ")
                fig = plt.figure()
                plt.plot(backtest[i], color="red")
                plt.plot(test_output[i], color="green")
                plt.savefig(self.path + "/backtest/" + "test" + str(i) + ".png")
                loop.update(1)
            with open(self.path + "/backtest/actual.npy", "wb") as f:
                np.save(f, test_output)
            with open(self.path + "/backtest/backtest.npy", "wb") as f:
                np.save(f, backtest)
        # save trained samples
        if input("Save trained samples? [yes/no]: ") == "yes":
            trained = np.array([normalize(result) for result in self.sess.run(self.layer[-1], feed_dict={self.input: training_input})])
            loop = tqdm.tqdm(total=trained.shape[0], position=0, leave=False)
            for i in range(trained.shape[0]):
                loop.set_description("Saving trained samples... ")
                fig = plt.figure()
                plt.plot(trained[i], color="red")
                plt.plot(training_output[i], color="green")
                plt.savefig(self.path + "/trained-samples/sample" + str(i) + ".png")
                loop.update(1)
    def run(self, data:list):
        results = [normalize(result) for result in self.sess.run(self.layer[-1], feed_dict={self.input: data})]
        return results

