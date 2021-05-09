
import tqdm as tqdm
import os, logging, json, ast
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np

# suppress warnings from TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
tf.disable_v2_behavior()

class DeepNeuralNetwork:
    def __init__(self, path="", hyper={}):
        """ hyper = {"architecture", "activation", "abs_synapse", "learning_rate"} """
        self.path = path
        if not self.load(): # if the model exists, self.load() automatically loads hyperparameters that are saved
            self.hyper = hyper # if the model doesn't exist, load given hyperparameters
        # load hyperparameters
        architecture  = self.hyper["architecture"]
        activation    = self.hyper["activation"]
        abs_synapse   = self.hyper["abs_synapse"]
        learning_rate = self.hyper["learning_rate"]
        with tf.Graph().as_default():
            # setup architecture (weights and layers w/ neurons)
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
    def save(self):
        self.saver.save(self.sess, self.path + "/dnn/checkpoint.ckpt", global_step=self.global_step)
        with open(self.path + "/dnn/hyperparameters", "w+") as f:
            f.write(json.dumps(self.hyper))
    def load(self):
        try:
            with open(self.path + "/dnn/hyperparameters", "r") as f:
                self.hyper = ast.literal_eval(f.read())
            return True
        except FileNotFoundError:
            return False
    def train(self, dataset={"input": [], "output": []}, iteration=int(), test=0.00):
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
        self.save()
        if test != 0.00:
            print("BACKTEST COST = ", self.sess.run(self.cost, feed_dict={self.input: test_input, self.output: test_output}))
            backtest = self.sess.run(self.layer[-1], feed_dict={self.input: test_input})
            # save backtesting samples
            loop = tqdm.tqdm(total=backtest.shape[0], position=0, leave=False)
            for i in range(backtest.shape[0]):
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
            results = self.sess.run(self.layer[-1], feed_dict={self.input: training_input})
            loop = tqdm.tqdm(total=results.shape[0], position=0, leave=False)
            for i in range(results.shape[0]):
                loop.set_description("Saving trained samples... ")
                fig = plt.figure()
                plt.plot(results[i], color="red")
                plt.plot(training_output[i], color="green")
                plt.savefig(self.path + "/trained-samples/sample" + str(i) + ".png")
                loop.update(1)
    def run(self, data=[]):
        results = self.sess.run(self.layer[-1], feed_dict={self.input: data})
        return results
