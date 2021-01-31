
import os, logging, json, ast
import tensorflow.compat.v1 as tf
import numpy as np

# suppress warnings from TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
tf.disable_v2_behavior()

class DeepNeuralNetwork:
    def __init__(self, path="", hyper={}):
        """ hyper = {"architecture", "activation", "abs_synapse", "learning_rate"} """
        self.path = path + "/dnn"
        if self.load() == False: # if model does not exist
            # if the model exists, self.load() automatically loads hyperparameters that are saved
            self.hyper = hyper # load given hyperparameters
        # load hyperparameters
        architecture  = hyper["architecture"]
        activation    = hyper["activation"]  # relu is highly recommended
        abs_synapse   = hyper["abs_synapse"]
        learning_rate = hyper["learning_rate"]
        # setup hidden layers
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.layer = [] # values in hidden layers
        self.input, self.output = tf.placeholder(tf.float32), tf.placeholder(tf.float32) # dataset
        self.weights = [tf.Variable(tf.random_uniform([layer[0], layer[1]], -abs_synapse, abs_synapse)) for layer in architecture]
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
        ckpt = tf.train.get_checkpoint_state(self.path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path) # load existing sessions saved in model directory
        else:
            self.sess.run(tf.global_variables_initializer())
    def save(self):
        self.saver.save(self.sess, self.path + "/checkpoint.ckpt", global_step=self.global_step)
        with open(self.path + "/hyperparameters", "w+") as f:
            f.write(json.dumps(self.hyper))
    def load(self):
        try:
            with open(self.path + "/hyperparameters", "r") as f:
                self.hyper = ast.literal_eval(f.read())
        except FileNotFoundError:
            return False
    def train(self, training_input=[], training_output=[], iteration=int(), test=0.00):
        if test != 0.00:
            total_samples = training_input.shape[0]
            test_input = training_input[-int(total_samples * test):]
            test_output = training_output[-int(total_samples * test):]
            training_input = training_input[:-int(total_samples * test)]
            training_output = training_output[:-int(total_samples * test)]
        for i in range(iteration):
            self.sess.run(self.model, feed_dict={self.input: training_input, self.output: training_output})
            if i % (iteration / 10) == 0:
                cost = self.sess.run(self.cost, feed_dict={self.input: training_input, self.output: training_output})
                print("ITERATION #{}: COST = {}" .format(i, cost))
        self.save()

